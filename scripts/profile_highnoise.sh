#!/usr/bin/env bash
# Capture an Nsight Systems trace of one high-noise diffusion step.
# Output: traces/highnoise.nsys-rep
set -euo pipefail
cd "$(dirname "$0")/.."

nsys profile \
  -o traces/highnoise \
  -t cuda,nvtx,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  python -m wan.test_high_noise --nsys "$@"
