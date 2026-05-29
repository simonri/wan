export TORCHINDUCTOR_CACHE_DIR="$PWD/.torchinductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run -m wan.runtime.launch_server
