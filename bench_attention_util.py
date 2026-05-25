"""Attention *utilization* (MFU) benchmark — how well each backend uses the GPU.

Reports achieved TFLOP/s and MFU for the three backends on the real Wan2.2 DiT
attention shapes (heads/dim/dtype + actual 720p/480p token counts):
  - FA4   : this repo's flash_attn.cute  (flash_attn_varlen_func_op)
  - FA3   : flash_attn_interface.flash_attn_func  (Hopper build)
  - SDPA  : torch scaled_dot_product_attention (auto-selected backend)

MFU is shown against two denominators:
  - GEMM peak : the GPU's *achievable* bf16 matmul throughput, measured live here.
                MFU-vs-GEMM is the honest "how close to pure-matmul" number; the gap
                is the softmax tax + memory/occupancy overhead.
  - peak      : the datasheet *dense* bf16 tensor-core peak (theoretical ceiling).

Run:  uv run python bench_attention_util.py
"""

import torch
import torch.nn.functional as F

from wan.configs.pipeline.wan import WanI2VConfig
from wan.layers.attention.flash_attn import flash_attn_varlen_func_op
from wan.torch_utils import PRECISION_TO_TYPE

try:
    import flash_attn_interface as _fa3
except Exception:
    _fa3 = None

# datasheet DENSE bf16 tensor-core peak (TFLOP/s), best-effort by device-name substring
PEAK_BF16 = {
    "H100 PCIe": 756.0, "H100 NVL": 835.0, "H100": 989.0, "H200": 989.0,
    "A100": 312.0, "L40S": 366.0, "L40": 181.0, "RTX 4090": 165.0, "RTX 6000 Ada": 182.0,
}


def bench_ms(fn, iters=20, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def latent_tokens(h, w, frames, patch=(1, 2, 2), spatial=8, temporal=4):
    lf = (frames - 1) // temporal + 1
    return (lf // patch[0]) * ((h // spatial) // patch[1]) * ((w // spatial) // patch[2])


def attn_flops(heads, nq, nkv, d):
    return 4 * heads * nq * nkv * d  # QKᵀ + PV, each 2·n²·d ops


def measure_gemm_peak(dtype, dev):
    """Achievable bf16 matmul TFLOP/s on this GPU (square GEMMs, take the best)."""
    best = 0.0
    for s in (8192, 16384):
        a = torch.randn(s, s, device=dev, dtype=dtype)
        b = torch.randn(s, s, device=dev, dtype=dtype)
        ms = bench_ms(lambda: torch.mm(a, b), iters=30, warmup=10)
        best = max(best, (2 * s ** 3) / (ms * 1e-3) / 1e12)
        del a, b
        torch.cuda.empty_cache()
    return best


def fa4_call(q, k, v, scale):
    return flash_attn_varlen_func_op(
        q=q, k=k, v=v, cu_seqlens_q=None, cu_seqlens_k=None,
        max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1],
        softmax_scale=scale, causal=False, return_softmax_lse=False,
    )


def fa3_call(q, k, v, scale):
    o = _fa3.flash_attn_func(q, k, v, softmax_scale=scale, causal=False)
    return o[0] if isinstance(o, tuple) else o


def sdpa_call(q, k, v, scale):
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    return F.scaled_dot_product_attention(qt, kt, vt, scale=scale).transpose(1, 2)


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    cfg = WanI2VConfig()
    arch = cfg.dit_config.arch_config
    H, D = arch.num_attention_heads, arch.attention_head_dim
    dtype = PRECISION_TO_TYPE[cfg.dit_precision]
    scale = D ** -0.5

    peak = next((v for kname, v in PEAK_BF16.items() if kname in name), None)
    gemm_peak = measure_gemm_peak(dtype, dev)
    print(f"GPU: {name}  dtype={dtype}  heads={H}  head_dim={D}")
    print(f"achievable bf16 GEMM peak (measured): {gemm_peak:.0f} TFLOP/s"
          + (f"   |   datasheet dense peak: {peak:.0f} TFLOP/s" if peak else "   |   datasheet peak: unknown"))
    print(f"FA3 available: {'yes' if _fa3 else 'NO'}\n")

    backends = [("FA4", fa4_call)]
    if _fa3 is not None:
        backends.append(("FA3", fa3_call))
    backends.append(("SDPA", sdpa_call))

    hdr = f"{'shape':<14}{'backend':<8}{'ms':>9}{'TFLOP/s':>10}{'MFU(GEMM)':>11}" + ("{:>10}".format("MFU(peak)") if peak else "")
    print(hdr); print("-" * len(hdr))
    for label, h, w, f in [("720p", 1280, 720, 81), ("480p", 832, 480, 81)]:
        N = latent_tokens(h, w, f)
        for kind, nkv in (("self", N), ("cross", arch.text_len)):
            q = torch.randn(1, N, H, D, device=dev, dtype=dtype)
            k = torch.randn(1, nkv, H, D, device=dev, dtype=dtype)
            v = torch.randn(1, nkv, H, D, device=dev, dtype=dtype)
            flops = attn_flops(H, N, nkv, D)
            for bname, fn in backends:
                try:
                    ms = bench_ms(lambda fn=fn: fn(q, k, v, scale))
                except Exception as exc:
                    print(f"{label + ' ' + kind:<14}{bname:<8}  n/a ({type(exc).__name__})")
                    continue
                tf = flops / (ms * 1e-3) / 1e12
                row = f"{label + ' ' + kind:<14}{bname:<8}{ms:>9.3f}{tf:>10.0f}{100 * tf / gemm_peak:>10.0f}%"
                if peak:
                    row += f"{100 * tf / peak:>9.0f}%"
                print(row)
            del q, k, v
            torch.cuda.empty_cache()
        print()

    print("MFU(GEMM) = achieved ÷ measured matmul peak (the softmax-tax-free ceiling on this GPU).")
    print("MFU(peak) = achieved ÷ datasheet dense bf16 peak (theoretical, never reachable).")


if __name__ == "__main__":
    main()
