# Qwen3.8-Flash-Next → W8A16 for vLLM 0.34 on 8× A100

`Qwen/Qwen3.8-Flash-Next` is quantized to weight-only int8 with llm-compressor (model-free PTQ). It is served by vLLM 0.34.0 on 8× A100 40GB (sm80).

## Layout

| Path | Purpose |
|---|---|
| `compressor.py` | Generic CLI with one profile per architecture (`PROFILES`) |
| `qwen3.8-flash-next-w8a16.sh`, `qwen3.8-flash-next-awq-w4a16.sh` | Quantize the local BF16 checkpoint (`SOURCE`, default `/mnt/data2/models/Qwen3.8-Flash-Next`) to `OUTPUT`. The AWQ script currently stops at the AWQ mapping check. |
| `compose.yaml` | vLLM OpenAI server (`vllm/vllm-openai:v0.34.0`) |
| `pyproject.toml`, `uv.lock` | uv project with pinned llm-compressor stack; dev group: pytest |
| `tests/test_compressor.py` | End-to-end tests on a tiny random `qwen4_exp` in the official checkpoint layout |
| `tests/vllm_smoke.py` | Load and generate in vLLM 0.34 (PEP 723 script with its own environment) |

## Usage

```bash
uv sync
./qwen3.8-flash-next-w8a16.sh --dry-run   # plan from safetensors headers, seconds
./qwen3.8-flash-next-w8a16.sh             # same as: uv run compressor.py SOURCE OUTPUT --scheme W8A16
MODEL_DIR=/mnt/data2/Qwen3.8-Flash-Next-W8A16 docker compose up -d
uv run pytest
```

`compressor.py` takes a **local** BF16 checkpoint directory. For the rare Hub case, run `uv run hf download MODEL --local-dir DIR` first.

Options (the profile is picked by the checkpoint's `model_type`; unknown types use `default`):
- `--scheme`: the int schemes that Marlin serves on sm80.
  - `W8A16` (default): int8 with one scale per output channel, data-free round-to-nearest on all GPUs via `model_free_ptq`. This differs from the compressed-tensors `W8A16` preset (group 128), which does not fit MoE experts at TP8.
  - `W4A16`: int4 with group size 128. It always runs AWQ (llm-compressor `oneshot` with calibration data). Before loading anything, a check on a weightless (meta) model stops the run unless AWQ mappings cover every int4 module. For `qwen4_exp` they do not: hyper-connections replace the usual pre-norms in front of the experts.
- `--dry-run`: print the plan (and the AWQ check) only.

**New architecture:** add an entry to `PROFILES`, run `--dry-run`, and check the printed module list.

## Model and decisions

- `qwen4_exp`: 180B params = 125B (6B active) + 51B n-gram embedding + 4B MTP. It has 48 layers (36 Gated DeltaNet + 12 QSA) and 512 routed experts (top-10, intermediate 640). Experts are stored fused: `gate_up_proj [512, 1280, 2560]`, `down_proj [512, 2560, 640]`.
- **Only routed experts are quantized** (123.3B params, 75,264 matrices; 360 GB → 236.9 GB). Same split as Qwen's FP8 release.
  - The n-gram embedding must stay BF16: vLLM's PLE layer accepts only BF16, FP8 or NVFP4 tables.
- **int8 per channel, not the `W8A16` preset.**
  - The preset uses group 128. At TP=8 each GPU holds 640/8 = 80 expert channels, and vLLM's Marlin and Triton MoE kernels reject that.
- **Target regex `re:.*\.mlp\.experts(\.|$)`** matches llm-compressor's split names and vLLM's fused layer name.
- **RTN, no calibration.** On the tiny model GPTQ gave the same KL (1.03e-6 vs 1.05e-6). GPTQ via transformers would also need ~95 GiB RAM for the n-gram table and would drop MTP.

## Serving (compose.yaml)

| Setting | Why |
|---|---|
| `--tensor-parallel-size 8 --enable-expert-parallel` | 64 whole experts per GPU. Plain TP8 makes Marlin pad 80 → 128 channels: 1.6× expert VRAM (~26 vs ~17 GB per GPU). |
| `--kv-cache-dtype fp8` | Halves the main KV cache. vLLM's QSA kernel has an sm80 path. |
| `--attention-config.indexer_kv_dtype bf16` | Set explicitly. The indexer's FP8 path needs FP8 dot products (sm89+), and its raw keys stay BF16 anyway. |
| `--max-model-len 102400` | 100k context per user. |
| `--max-num-seqs 12` | ≈15 GB KV per GPU ÷ ≈1.2 GB per 100k sequence (13 attention layers incl. MTP at ≈11 KB/token, plus 4 × 14 MB GDN state for MTP). Without MTP it would be 16. Verify against "Maximum concurrency" in the vLLM startup log. |
| `--max-num-batched-tokens 8192` | vLLM's A100 default is 2048. That splits a 100k prompt into 50 chunks, and MTP warns below 8192. Trade-off: faster prefill, but longer pauses for other users during a prefill chunk. |
| `--long-prefill-token-threshold 4096` | While other requests wait, a long prompt takes at most half of each step's token budget. A pasted 100k document no longer stalls the other users. |
| `--enable-prompt-tokens-details`, `--enable-per-request-metrics` | Responses include cached prompt tokens and per-request timings (TTFT, ITL, tokens/s, MTP stats). `--enable-force-include-usage` is not implemented in the Rust frontend and is not needed behind agentgateway: for LLM backends, agentgateway forces `stream_options.include_usage` itself, and the Rust frontend answers with a final usage chunk (tested). |
| `--speculative-config {"method":"mtp","num_speculative_tokens":3}` | Recipe setting: the built-in MTP layer drafts 3 tokens per step. |
| `--performance-mode interactivity` | CUDA graphs for every batch size up to 32, which fits a few long requests. Compilation is already maximal by default (`-O2`; `-O3` is identical in 0.34). |
| `--mm-encoder-tp-mode data` | The vision encoder runs whole on each GPU instead of split 8 ways (+0.8 GB per GPU). If no images are needed, use `--language-model-only` instead. |
| `--engram-config {"cpu_offload":true}` + host RAM ≈ 110 GB | vLLM pins the 102 GB BF16 n-gram table in host memory (default, made explicit). `use_thp` only affects DeepSeek-V4.1 in 0.34. |
| `VLLM_USE_RUST_FRONTEND=1` | Rust HTTP, tokenizer and parser frontend (recipe). It ignores `--enable-auto-tool-choice` (tool choice is automatic), so that flag is not set. |

Already on by default, so not set: async scheduling and chunked prefill. Tool calling and reasoning use the recipe parsers `qwen3_coder` and `qwen3`.

## Validation

| Check | Result |
|---|---|
| Dry run on the real model | 75,264 matrices validated on meta tensors, no weights downloaded |
| Real shard (layer 10 `gate_up_proj`, 1024 matrices) | Relative weight error median 0.89 %, max 1.59 %. 4.4 s on GPU, 6.5 GB RAM, 5.1 GB VRAM |
| Logits BF16 vs. dequantized (pytest) | KL/token ≈ 1e-6. Swapped gate/up raises KL >10× (control) |
| vLLM 0.34.0 on the tiny model | Picks `CompressedTensorsWNA16MoEMethod` + Marlin MoE. n-gram table loads BF16 and pinned. Mean \|Δlogprob\| vs. BF16 0.0025 |
| `compose.yaml` | `docker compose config` OK. All flags were run with `vllm serve` 0.34 on the tiny W8A16 model, scaled to TP1 and a 4k context. Verified: Rust frontend, MTP drafter with W8A16 experts, Marlin MoE, FP8 KV, BF16 indexer, encoder DP, pinned n-gram table, and a tool request answered with HTTP 200. |
| W4A16 + AWQ (pytest, tiny Qwen3) | AWQ runs and folds its scales into the norms. |
| W4A16 guard on the real model (`--dry-run`, ~40 s) | Checked on a weightless (meta) model before any loading. For `qwen4_exp` the built-in AWQ mappings miss 49,152 of 73,728 int4 modules (the expert `gate_proj`/`up_proj`), so the run stops. |

**Not tested:**
- Real 8× A100 hardware: the sm80 paths are verified from source only, and tests ran on one RTX 5070 Ti (sm120), which uses different FP8 and CuTe paths.
- TP8 + EP across GPUs.
- MTP acceptance rate: the tiny model's random weights cannot measure it.
- Quality on real benchmarks. Run lm-eval after quantization.

vLLM 0.34 strict-checks **unquantized** `qwen4_exp` checkpoints for a hyper-connection padding shard that no checkpoint contains. Workaround: `--model-loader-extra-config '{"enable_weights_track": false}'`.

## References

- Model: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next> (FP8 split reference: `Qwen/Qwen3.8-Flash-Next-FP8`)
- vLLM recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>
- llm-compressor: <https://github.com/vllm-project/llm-compressor>
- vLLM 0.34.0 source: `vllm/models/qwen4_exp/`, `fused_moe/oracle/int_wna16.py`, `quantization/utils/marlin_utils.py`
