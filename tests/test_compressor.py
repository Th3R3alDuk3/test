"""End-to-end tests on a tiny random Qwen3.8-Flash-Next (qwen4_exp).

The tiny checkpoint uses the official layout (fused 3D experts, sharded n-gram embedding).
Expert intermediate size 80 = what each GPU holds of the real model at TP=8.
"""

import json
import re
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from huggingface_hub import hf_hub_download
from llmcompressor.entrypoints.model_free import model_free_ptq
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import compressor

REPO = "Qwen/Qwen3.8-Flash-Next"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> Path:
    cfg = json.loads(Path(hf_hub_download(REPO, "config.json")).read_text())
    cfg.pop("transformers_version")
    cfg["text_config"].update(  # head_dim stays 256: rotary dim must match mrope_section
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=80,
        shared_expert_intermediate_size=80,
        hc_lowrank=32,
        indexer_head_dim=64,
        ple_embed_dim=256,
        ple_layer_ids=[2],
        ngram_vocab_size_base=1000,
        split_ngram_parts=8,
    )
    cfg["vision_config"].update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256)

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**cfg), dtype=torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path, max_shard_size="60MB")

    # transformers saves experts one by one; the official checkpoint fuses them:
    # gate_up_proj [E, 2I, H] (gate first), down_proj [E, H, I]
    index_file = path / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    index["weight_map"] = {}
    for shard in sorted(path.glob("*.safetensors")):
        tensors = load_file(shard)
        experts = {}
        for name in [n for n in tensors if re.search(r"\.experts\.\d+\.", n)]:
            base, expert, proj = re.fullmatch(r"(.*\.experts)\.(\d+)\.(\w+)_proj\.weight", name).groups()
            experts.setdefault(base, {}).setdefault(int(expert), {})[proj] = tensors.pop(name)
        for base, by_index in experts.items():
            ordered = [by_index[i] for i in sorted(by_index)]
            tensors[f"{base}.gate_up_proj"] = torch.stack([torch.cat([e["gate"], e["up"]]) for e in ordered])
            tensors[f"{base}.down_proj"] = torch.stack([e["down"] for e in ordered])
        save_file(tensors, shard, metadata={"format": "pt"})
        index["weight_map"].update(dict.fromkeys(tensors, shard.name))

    # transformers drops the MTP head; rebuild it like the official checkpoint:
    # a full-attention decoder layer (here a copy of the last one) plus projections and a mixer
    tensors = {k: v for f in path.glob("*.safetensors") for k, v in load_file(f).items()}
    last, hidden, streams = "model.language_model.layers.3.", 256, 4 * 256
    mtp = {"mtp.layers.0." + k.removeprefix(last): v for k, v in tensors.items() if k.startswith(last)}
    mtp |= {k.replace("model.language_model.", "mtp."): v for k, v in tensors.items() if "hyper_connection_mixer" in k}
    mtp |= {
        "mtp.fc_embedding.weight": (torch.randn(hidden, hidden) * 0.02).bfloat16(),
        "mtp.fc_hidden.weight": (torch.randn(hidden, hidden) * 0.02).bfloat16(),
        "mtp.pre_fc_norm_embedding.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "mtp.pre_fc_norm_hidden.weight": torch.ones(streams, dtype=torch.bfloat16),
    }
    save_file(mtp, path / "model-mtp.safetensors", metadata={"format": "pt"})
    index["weight_map"].update(dict.fromkeys(mtp, "model-mtp.safetensors"))
    index_file.write_text(json.dumps(index))

    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ]:
        shutil.copyfile(hf_hub_download(REPO, name), path / name)
    return path


@pytest.fixture(scope="session")
def config():
    return compressor.quant_config(compressor.PROFILES["qwen4_exp"], "W8A16")


@pytest.fixture(scope="session")
def quantized(tiny_model, config, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("w8a16")
    model_free_ptq(tiny_model, out, config=config)
    return out


def test_plan_matches_output(tiny_model, config, quantized):
    planned, _, _ = compressor.plan(tiny_model, config)
    weight_map = json.loads((quantized / "model.safetensors.index.json").read_text())["weight_map"]
    layers, experts, projections = 4 + 1, 8, 3  # decoder layers + MTP layer
    assert planned.total() == sum(k.endswith(".weight_scale") for k in weight_map) == layers * experts * projections


@torch.no_grad()
def test_dequantized_logits_match(tiny_model, quantized):
    model = AutoModelForImageTextToText.from_pretrained(tiny_model, dtype=torch.bfloat16).to(DEVICE).eval()
    text = "The quick brown fox jumps over the lazy dog. " * 8
    ids = AutoTokenizer.from_pretrained(tiny_model)(text, return_tensors="pt").input_ids.to(DEVICE)
    reference = model(input_ids=ids).logits.float().log_softmax(-1)
    packed = {k: v for f in quantized.glob("*.safetensors") for k, v in load_file(f).items()}

    def dequantize(module: str) -> torch.Tensor:  # int8 per channel: q * scale
        shape = torch.Size(packed[f"{module}.weight_shape"].tolist())
        return unpack_from_int32(packed[f"{module}.weight_packed"], 8, shape) * packed[f"{module}.weight_scale"]

    kl = {}
    for gate, up in [("gate", "up"), ("up", "gate")]:  # second pass: gate/up swapped as negative control
        for name, param in model.named_parameters():
            if match := re.fullmatch(r"(.*\.experts)\.(gate_up_proj|down_proj)", name):
                base, kind = match.groups()
                if kind == "gate_up_proj":
                    experts = [
                        torch.cat([dequantize(f"{base}.{e}.{gate}_proj"), dequantize(f"{base}.{e}.{up}_proj")])
                        for e in range(param.shape[0])
                    ]
                else:
                    experts = [dequantize(f"{base}.{e}.down_proj") for e in range(param.shape[0])]
                param.copy_(torch.stack(experts))
        logits = model(input_ids=ids).logits.float().log_softmax(-1)
        kl[gate] = F.kl_div(logits, reference, log_target=True, reduction="sum").item() / ids.shape[1]

    assert kl["gate"] < 1e-5  # int8 per channel is close to lossless
    assert kl["up"] > 10 * kl["gate"]  # the check notices a layout mix-up


@pytest.fixture(scope="session")
def tiny_qwen3(tmp_path_factory) -> Path:
    """Tiny random Qwen3, an architecture with built-in AWQ mappings."""
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("qwen3")
    AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(path)
    return path


def test_w4a16_runs_awq(tiny_qwen3, tmp_path, monkeypatch):
    monkeypatch.setattr(compressor, "NUM_SAMPLES", 16)
    monkeypatch.setattr(compressor, "SEQ_LEN", 128)
    config = compressor.quant_config(compressor.PROFILES["default"], "W4A16")
    compressor.check_awq_mappings(tiny_qwen3, config)
    compressor.quantize_awq(tiny_qwen3, tmp_path, config)

    weights = {k: v for f in tmp_path.glob("*.safetensors") for k, v in load_file(f).items()}
    weights_args = json.loads((tmp_path / "config.json").read_text())["quantization_config"]["config_groups"]
    assert next(iter(weights_args.values()))["weights"]["num_bits"] == 4
    # AWQ folds its smoothing scales into the norms in front of the quantized layers
    original = load_file(next(tiny_qwen3.glob("*.safetensors")))
    norm = "model.layers.0.post_attention_layernorm.weight"
    assert not torch.equal(weights[norm], original[norm])


def test_w4a16_refuses_incomplete_awq_mappings(tiny_model):
    config = compressor.quant_config(compressor.PROFILES["qwen4_exp"], "W4A16")
    with pytest.raises(SystemExit, match=r"miss .*experts\.N\.gate_proj"):
        compressor.check_awq_mappings(tiny_model, config)
