"""Multi-backend attention benchmark for the Wan2.2 I2V DiT shapes.

Compares, on this GPU and the real config (heads/dim/dtype + actual token counts),
the attention backends you can pick from:
  - FA4(cute)   : flash_attn.cute via FlashAttentionImpl  (what the model uses now; Blackwell-first)
  - FA3(hopper) : flash_attn_interface.flash_attn_func     (Hopper-tuned; H100-first)
  - SDPA-cuDNN  : torch SDPA forced to the cuDNN backend   (NVIDIA Hopper fused attn)
  - SDPA-Flash  : torch SDPA forced to FlashAttention-2
Forward-only (inference), self- and cross-attention.

Run:  uv run python bench_attention.py
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from wan.configs.pipeline.wan import WanI2VConfig
from wan.layers.attention.flash_attn import FlashAttentionImpl
from wan.torch_utils import PRECISION_TO_TYPE

try:
    import flash_attn_interface as _fa3
except Exception:
    _fa3 = None


def latent_token_count(height, width, num_frames, patch=(1, 2, 2), spatial=8, temporal=4):
    lat_f = (num_frames - 1) // temporal + 1
    return (lat_f // patch[0]) * ((height // spatial) // patch[1]) * ((width // spatial) // patch[2])


def bench_ms(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def attn_tflops(h, nq, nkv, d, ms):
    return (2 * (2 * h * nq * nkv * d)) / (ms * 1e-3) / 1e12


def fa3_call(q, k, v, scale):
    o = _fa3.flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
    return o[0] if isinstance(o, tuple) else o


def sdpa_call(q, k, v, scale, backend):
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # [B,N,H,D] -> [B,H,N,D]
    with sdpa_kernel(backend):
        o = F.scaled_dot_product_attention(qt, kt, vt, scale=scale)
    return o.transpose(1, 2)


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = "cuda"
    cfg = WanI2VConfig()
    arch = cfg.dit_config.arch_config
    H, D = arch.num_attention_heads, arch.attention_head_dim
    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    scale = D ** -0.5
    text_len, n_layers = arch.text_len, arch.num_layers

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"dtype={dtype}  heads={H}  head_dim={D}  text_len={text_len}  layers={n_layers}  FA3={'yes' if _fa3 else 'NO'}\n")

    impl = FlashAttentionImpl(num_heads=H, head_size=D, causal=False, softmax_scale=scale)

    candidates = [("FA4(cute)", lambda q, k, v: impl.forward(q, k, v, None))]
    if _fa3 is not None:
        candidates.append(("FA3(hopper)", lambda q, k, v: fa3_call(q, k, v, scale)))
    candidates.append(("SDPA-cuDNN", lambda q, k, v: sdpa_call(q, k, v, scale, SDPBackend.CUDNN_ATTENTION)))
    candidates.append(("SDPA-Flash", lambda q, k, v: sdpa_call(q, k, v, scale, SDPBackend.FLASH_ATTENTION)))

    resolutions = [("720p", 1280, 720, 81), ("480p", 832, 480, 81)]
    best_per_block = {}  # label -> ms summed over (self+cross) using the fastest backend each

    for label, h, w, f in resolutions:
        N = latent_token_count(h, w, f)
        for kind, nkv in (("self", N), ("cross", text_len)):
            print(f"=== {label} {kind}  (N_q={N}, N_kv={nkv}) ===")
            q = torch.randn(1, N, H, D, device=dev, dtype=dtype)
            k = torch.randn(1, nkv, H, D, device=dev, dtype=dtype)
            v = torch.randn(1, nkv, H, D, device=dev, dtype=dtype)

            ref = impl.forward(q, k, v, None)  # FA4 as correctness reference
            fa4_ms = None
            best_ms = None
            for name, fn in candidates:
                try:
                    out = fn(q, k, v)
                    diff = out.sub(ref).abs().amax().item()
                    ms = bench_ms(lambda f=fn, qq=q, kk=k, vv=v: f(qq, kk, vv))
                except Exception as exc:
                    print(f"  {name:<13} n/a ({type(exc).__name__})")
                    continue
                if name == "FA4(cute)":
                    fa4_ms = ms
                best_ms = ms if best_ms is None else min(best_ms, ms)
                rel = f"{fa4_ms / ms:.2f}x vs FA4" if fa4_ms else ""
                print(f"  {name:<13}{ms:>9.3f} ms{attn_tflops(H, N, nkv, D, ms):>9.0f} TF/s   Δ={diff:.1e}   {rel}")
            best_per_block.setdefault(label, 0.0)
            best_per_block[label] += best_ms or 0.0
            del q, k, v, ref
            torch.cuda.empty_cache()
            print()

    steps = 8
    print(f"Projected attention-only time per generation ({n_layers} layers x {steps} steps, fastest backend each shape):")
    for label, ms in best_per_block.items():
        print(f"  {label}: {ms * n_layers * steps / 1000:.1f}s")


if __name__ == "__main__":
    main()
