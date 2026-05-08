#!/usr/bin/env bash
# Capture an Nsight Systems trace of one T5 encode pass.
# Output: traces/t5_encode.nsys-rep
set -euo pipefail
cd "$(dirname "$0")/.."

nsys profile \
  -o traces/t5_encode \
  -t cuda,nvtx,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  uv run python -m wan.test_t5 --nsys "$@"
