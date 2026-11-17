#!/usr/bin/env bash
# Qwen3.8-Flash-Next -> W4A16 with AWQ for vLLM on A100. Serve with --enable-expert-parallel:
# group 128 does not fit the 80 expert channels per GPU at plain TP8.
# Currently stops at the AWQ check (seconds, before loading): llm-compressor 0.15 has no AWQ
# mappings for the qwen4_exp experts. Calibration loads the model via transformers (~95 GiB RAM
# for the n-gram table, the rest offloads to disk).
set -euo pipefail
cd "$(dirname "$0")"

SOURCE=${SOURCE:-/mnt/data2/models/Qwen3.8-Flash-Next}
OUTPUT=${OUTPUT:-/mnt/data2/Qwen3.8-Flash-Next-AWQ-W4A16}

exec uv run compressor.py "$SOURCE" "$OUTPUT" --scheme W4A16 "$@"
