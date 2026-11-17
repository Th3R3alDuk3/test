from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from json import loads
from pathlib import Path
from re import sub
from sys import exit

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import (
    match_modules_set,
    match_named_modules,
    patch_attr,
)
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.entrypoints.model_free import model_free_ptq
from llmcompressor.entrypoints.model_free.converter import (
    ModelFreePtqConverter,
)
from llmcompressor.entrypoints.model_free.validate import validate_config
from llmcompressor.modeling.moe import linear_experts
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import (
    AWQModifier,
    get_layer_mappings_from_model,
)
from llmcompressor.utils import load_context
from safetensors import safe_open
from torch import bfloat16, device, empty, float32, int64
from torch.nn import Linear
from torch.utils._pytree import tree_leaves
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
)

DESCRIPTION = (
    "Quantizes a local BF16 checkpoint for vLLM on A100 (sm80): int weights, "
    "BF16 activations, Marlin kernels."
)


@dataclass(frozen=True)
class Profile:
    # compressed-tensors module names; "re:" marks a regex.
    targets: list[str]
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])


# Picked by the "model_type" in config.json; other types use "default".
PROFILES = {
    # Routed experts only, as in Qwen's FP8 release. The regex matches
    # llm-compressor's split experts (...mlp.experts.7.up_proj) and vLLM's
    # fused layer (...mlp.experts).
    "qwen4_exp": Profile(
        targets=[r"re:.*\.mlp\.experts(\.|$)"],
    ),
    "default": Profile(
        targets=["Linear"],
        ignore=[
            "lm_head",
            "re:.*embed.*",
            r"re:.*\.mlp\.gate$",
            "re:.*router.*",
            "re:.*shared_expert_gate$",
            "re:.*(visual|vision).*",
            "re:.*conv.*",
        ],
    ),
}

# AWQ calibration data, streamed from the Hub.
CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
CALIBRATION_SPLIT = "train_sft"
NUM_SAMPLES = 256
# Tokens per sample, at most.
SEQ_LEN = 512

# safetensors dtypes in BF16 checkpoints: BF16 weights, F32 and I64 buffers.
_DTYPES = {
    "BF16": bfloat16,
    "F32": float32,
    "I64": int64,
}


def quant_config(
    profile: Profile,
    scheme: str,
) -> QuantizationConfig:

    if scheme == "W8A16":
        # int8 with one scale per output channel. The compressed-tensors
        # "W8A16" preset uses group 128, which vLLM's Marlin MoE rejects once
        # TP splits experts below 128 channels (Qwen3.8 at TP8: 80).
        quant_scheme = QuantizationScheme(
            targets=profile.targets,
            weights=QuantizationArgs(
                num_bits=8,
                type="int",
                symmetric=True,
                strategy="channel",
            ),
        )
    else:
        # W4A16: int4, group 128.
        quant_scheme = preset_name_to_scheme("W4A16", profile.targets)

    return validate_config(None, quant_scheme, profile.ignore)


def plan(
    model_dir: Path,
    config: QuantizationConfig,
) -> tuple[Counter[str], int, int]:

    # Validates every shard on meta tensors. Returns the quantized layers per
    # name pattern and the bytes before and after.
    converter = ModelFreePtqConverter(config)
    quantized: Counter[str] = Counter()
    size_before = size_after = 0

    for shard_path in sorted(model_dir.glob("*.safetensors")):

        with safe_open(shard_path, "pt") as shard:
            # `safe_open` is not iterable.
            slices = {
                name: shard.get_slice(name)
                for name in shard.keys()  # noqa: SIM118
            }

        tensors = {
            name: empty(
                tensor_slice.get_shape(),
                dtype=_DTYPES[tensor_slice.get_dtype()],
                device="meta",
            )
            for name, tensor_slice in slices.items()
        }
        quantized_tensors = converter.validate(tensors)

        size_before += sum(tensor.nbytes for tensor in tensors.values())
        size_after += sum(
            tensor.nbytes for tensor in quantized_tensors.values())

        # One scale per quantized layer; indices become N.
        quantized.update(
            sub(r"\.\d+\.", ".N.", name.removesuffix(".weight_scale"))
            for name in quantized_tensors
            if name.endswith(".weight_scale")
        )

    return quantized, size_before, size_after


def _auto_model_class(
    model_dir: Path,
) -> type:

    model_config = AutoConfig.from_pretrained(model_dir)

    if hasattr(model_config, "vision_config"):
        return AutoModelForImageTextToText

    return AutoModelForCausalLM


def check_awq_mappings(
    model_dir: Path,
    config: QuantizationConfig,
) -> None:

    # Exits unless AWQ mappings cover every int4 layer; AWQ silently skips
    # mappings that match nothing, which would leave plain int4 rounding.
    # A weightless (meta) model takes seconds instead of a full load.
    with device("meta"):
        model = _auto_model_class(model_dir).from_config(
            AutoConfig.from_pretrained(model_dir),
        )

    # Meta tensors cannot be offloaded.
    with patch_attr(
        linear_experts,
        "offload_module",
        lambda *args, **kwargs: None,
    ):
        # The Linear expert layout that quantize_awq gets from load_context().
        linearize_moe(model)

    (scheme,) = config.config_groups.values()
    names = {module: name for name, module in model.named_modules()}

    # Targets can also match containers such as mlp.experts; only Linear
    # layers get quantized.
    int4_layers = {
        name
        for name, module in match_named_modules(
            model, scheme.targets, config.ignore)
        if isinstance(module, Linear)
    }
    balanced_layers: set[str] = set()

    for mapping in get_layer_mappings_from_model(model):

        try:
            matches = list(match_modules_set(
                model, (mapping.smooth_layer, *mapping.balance_layers)))
        except ValueError:
            # Mapping only partly present, e.g. q/k/v without input_layernorm.
            continue

        balanced_layers.update(
            names[layer]
            for _, *balance_layers in matches
            for layer in tree_leaves(balance_layers)
        )

    missing = int4_layers - balanced_layers

    if missing:
        patterns = sorted({
            sub(r"\.\d+(?=\.|$)", ".N", name)
            for name in missing
        })
        exit(
            f"AWQ mappings of {type(model).__name__} miss {len(missing)}/"
            f"{len(int4_layers)} layers: {patterns}"
        )

    print(f"AWQ mappings cover all {len(int4_layers)} int4 layers")


def quantize_awq(
    model_dir: Path,
    output_dir: Path,
    config: QuantizationConfig,
) -> None:

    model_class = _auto_model_class(model_dir)

    # Offloads to CPU and disk, and splits fused MoE experts into Linear layers.
    with load_context(model_class):
        model = model_class.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto_offload",
        )

    processor = AutoProcessor.from_pretrained(model_dir)
    # Multimodal processors wrap the tokenizer.
    tokenizer = getattr(processor, "tokenizer", processor)

    rows = load_dataset(
        CALIBRATION_DATASET,
        split=CALIBRATION_SPLIT,
        streaming=True,
    ).take(NUM_SAMPLES)
    texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False)
        for row in rows
    ]
    dataset = Dataset.from_dict(tokenizer(
        texts,
        max_length=SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    ))

    oneshot(
        model=model,
        processor=processor,
        dataset=dataset,
        recipe=[
            AWQModifier(),
            QuantizationModifier(
                config_groups=config.config_groups,
                ignore=config.ignore,
            ),
        ],
        max_seq_length=SEQ_LEN,
        num_calibration_samples=NUM_SAMPLES,
    )

    model.save_pretrained(output_dir, save_compressed=True)
    processor.save_pretrained(output_dir)


def main() -> None:

    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Local BF16 checkpoint directory.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for the quantized checkpoint.",
    )
    parser.add_argument(
        "--scheme",
        choices=["W8A16", "W4A16"],
        default="W8A16",
        help=(
            "W8A16: data-free round-to-nearest on the safetensors files, on "
            "all GPUs. W4A16: AWQ calibration, then int4; loads the model."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan, write nothing.",
    )
    args = parser.parse_args()

    config_path = args.model_dir / "config.json"
    model_type = loads(config_path.read_text())["model_type"]
    profile_name = model_type if model_type in PROFILES else "default"
    config = quant_config(PROFILES[profile_name], args.scheme)

    quantized, size_before, size_after = plan(args.model_dir, config)

    for pattern, count in quantized.most_common():
        print(f"{count:>8}  {pattern}")

    print(
        f"profile {profile_name}: {quantized.total()} layers, "
        f"{size_before / 1e9:.1f} -> {size_after / 1e9:.1f} GB"
    )

    # Before any loading, so a gap costs seconds.
    if args.scheme == "W4A16":
        check_awq_mappings(args.model_dir, config)

    if args.dry_run:
        return

    if args.scheme == "W4A16":
        quantize_awq(args.model_dir, args.output_dir, config)
    else:
        model_free_ptq(args.model_dir, args.output_dir, config=config)


if __name__ == "__main__":
    main()
