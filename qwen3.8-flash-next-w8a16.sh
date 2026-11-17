#!/usr/bin/env bash
# Qwen3.8-Flash-Next -> W8A16 (int8 per channel, routed experts only) for vLLM on A100.
set -euo pipefail
cd "$(dirname "$0")"

SOURCE=${SOURCE:-/mnt/data2/models/Qwen3.8-Flash-Next}
OUTPUT=${OUTPUT:-/mnt/data2/Qwen3.8-Flash-Next-W8A16}

exec uv run compressor.py "$SOURCE" "$OUTPUT" --scheme W8A16 "$@"
