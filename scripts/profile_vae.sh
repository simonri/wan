#!/usr/bin/env bash
# Capture an Nsight Systems trace of one VAE encode pass.
# Output: traces/vae_encode.nsys-rep
set -euo pipefail
cd "$(dirname "$0")/.."

nsys profile \
  -o traces/vae_encode \
  -t cuda,nvtx,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  uv run python -m wan.test_vae --nsys "$@"
