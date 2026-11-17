"""Quantize a local BF16 checkpoint for vLLM on A100 (sm80): int weights, BF16 activations, Marlin kernels.

    uv run compressor.py MODEL_DIR OUTPUT_DIR [--scheme W8A16|W4A16] [--dry-run]

W8A16: data-free round-to-nearest on the safetensors files, on all GPUs.
W4A16: AWQ calibration, then int4; loads the model.
PROFILES picks the quantized layers by config.json "model_type".
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    preset_name_to_scheme,
)
from compressed_tensors.utils import match_modules_set, match_named_modules, patch_attr
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.entrypoints.model_free import model_free_ptq
from llmcompressor.entrypoints.model_free.converter import ModelFreePtqConverter
from llmcompressor.entrypoints.model_free.validate import validate_config
from llmcompressor.modeling.moe import linear_experts
from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier, get_layer_mappings_from_model
from llmcompressor.utils import load_context
from safetensors import safe_open
from torch.utils._pytree import tree_leaves
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor


@dataclass(frozen=True)
class Profile:
    targets: list[str]  # compressed-tensors module names, "re:" prefix for regex
    ignore: list[str] = field(default_factory=lambda: ["lm_head"])


PROFILES = {
    # Routed experts only, as in Qwen's FP8 release. The regex matches llm-compressor's
    # split experts (...mlp.experts.7.up_proj) and vLLM's fused layer (...mlp.experts).
    "qwen4_exp": Profile(targets=[r"re:.*\.mlp\.experts(\.|$)"]),
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

# AWQ calibration data, streamed from the Hub
CALIBRATION_DATASET, CALIBRATION_SPLIT = "HuggingFaceH4/ultrachat_200k", "train_sft"
NUM_SAMPLES, SEQ_LEN = 256, 512

# safetensors dtypes found in BF16 checkpoints (BF16 weights, F32/I64 buffers)
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64}


def quant_config(profile: Profile, scheme: str) -> QuantizationConfig:
    if scheme == "W8A16":
        # int8 with one scale per output channel. The compressed-tensors "W8A16" preset uses group 128,
        # which vLLM's Marlin MoE rejects once TP splits experts below 128 channels (Qwen3.8 at TP8: 80).
        weights = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="channel")
        quant_scheme = QuantizationScheme(targets=profile.targets, weights=weights)
    else:  # W4A16: int4, group 128
        quant_scheme = preset_name_to_scheme("W4A16", profile.targets)
    return validate_config(None, quant_scheme, profile.ignore)


def plan(model: Path, config: QuantizationConfig) -> tuple[Counter, int, int]:
    """Validate every shard on meta tensors; return quantized modules per name pattern and bytes before/after."""
    converter = ModelFreePtqConverter(config)
    quantized, size_in, size_out = Counter(), 0, 0
    for path in sorted(model.glob("*.safetensors")):
        with safe_open(path, "pt") as f:
            slices = {name: f.get_slice(name) for name in f.keys()}  # noqa: SIM118 - safe_open is not iterable
            tensors = {
                name: torch.empty(s.get_shape(), dtype=DTYPES[s.get_dtype()], device="meta")
                for name, s in slices.items()
            }
        out = converter.validate(tensors)
        size_in += sum(t.nbytes for t in tensors.values())
        size_out += sum(t.nbytes for t in out.values())
        scales = [k.removesuffix(".weight_scale") for k in out if k.endswith(".weight_scale")]
        quantized.update(re.sub(r"\.\d+\.", ".N.", name) for name in scales)
    return quantized, size_in, size_out


def auto_model_class(model: Path) -> type:
    has_vision = hasattr(AutoConfig.from_pretrained(model), "vision_config")
    return AutoModelForImageTextToText if has_vision else AutoModelForCausalLM


def check_awq_mappings(model: Path, config: QuantizationConfig):
    """Exit unless AWQ mappings cover every int4 layer.

    AWQ silently skips mappings that match nothing, which would leave plain int4 rounding.
    Runs on a weightless (meta) model, so it takes seconds instead of a full load.
    """
    with torch.device("meta"):
        net = auto_model_class(model).from_config(AutoConfig.from_pretrained(model))
    with patch_attr(linear_experts, "offload_module", lambda *args, **kwargs: None):  # meta can't be offloaded
        linearize_moe(net)  # the Linear expert layout that quantize_awq gets from load_context()

    (scheme,) = config.config_groups.values()
    names = {module: name for name, module in net.named_modules()}
    int4 = {  # targets can also match containers such as mlp.experts; only Linear layers get quantized
        name
        for name, module in match_named_modules(net, scheme.targets, config.ignore)
        if isinstance(module, torch.nn.Linear)
    }
    balanced = set()
    for mapping in get_layer_mappings_from_model(net):
        try:
            matches = list(match_modules_set(net, (mapping.smooth_layer, *mapping.balance_layers)))
        except ValueError:  # mapping only partly present, e.g. q/k/v without input_layernorm
            continue
        balanced.update(names[layer] for _, *balance in matches for layer in tree_leaves(balance))
    if missing := int4 - balanced:
        patterns = sorted({re.sub(r"\.\d+(?=\.|$)", ".N", name) for name in missing})
        raise SystemExit(f"AWQ mappings of {type(net).__name__} miss {len(missing)}/{len(int4)} layers: {patterns}")
    print(f"AWQ mappings cover all {len(int4)} int4 layers")


def quantize_awq(model: Path, out: Path, config: QuantizationConfig):
    model_cls = auto_model_class(model)
    with load_context(model_cls):  # offloads to CPU/disk and splits fused MoE experts into Linear layers
        net = model_cls.from_pretrained(model, dtype="auto", device_map="auto_offload")
    processor = AutoProcessor.from_pretrained(model)
    tokenizer = getattr(processor, "tokenizer", processor)

    rows = load_dataset(CALIBRATION_DATASET, split=CALIBRATION_SPLIT, streaming=True).take(NUM_SAMPLES)
    texts = [tokenizer.apply_chat_template(row["messages"], tokenize=False) for row in rows]
    dataset = Dataset.from_dict(tokenizer(texts, max_length=SEQ_LEN, truncation=True, add_special_tokens=False))
    oneshot(
        model=net,
        processor=processor,
        dataset=dataset,
        recipe=[AWQModifier(), QuantizationModifier(config_groups=config.config_groups, ignore=config.ignore)],
        max_seq_length=SEQ_LEN,
        num_calibration_samples=NUM_SAMPLES,
    )
    net.save_pretrained(out, save_compressed=True)
    processor.save_pretrained(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="local BF16 checkpoint directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--scheme", choices=["W8A16", "W4A16"], default="W8A16")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args()

    model_type = json.loads((args.model / "config.json").read_text())["model_type"]
    profile = model_type if model_type in PROFILES else "default"
    config = quant_config(PROFILES[profile], args.scheme)

    quantized, size_in, size_out = plan(args.model, config)
    for pattern, count in quantized.most_common():
        print(f"{count:>8}  {pattern}")
    print(f"profile {profile}: {quantized.total()} layers, {size_in / 1e9:.1f} -> {size_out / 1e9:.1f} GB")
    if args.scheme == "W4A16":
        check_awq_mappings(args.model, config)
    if args.dry_run:
        return

    if args.scheme == "W4A16":
        quantize_awq(args.model, args.output, config)
    else:
        model_free_ptq(args.model, args.output, config=config)


if __name__ == "__main__":
    main()
