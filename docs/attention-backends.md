# Attention: cost, backends, and utilization

Where the DiT's time goes, which attention kernel is fastest on our hardware, and how
close to the hardware ceiling it runs. **TL;DR: attention is ~71% of the forward and is
already running at ~78% of this GPU's achievable matmul peak with FlashAttention‑4 — so
the kernel is near-optimal; the only real speedup lever is doing *fewer* attention FLOPs
(sparse attention / token merge / lower resolution), not a faster kernel or `torch.compile`.**

Measurements: NVIDIA **H100 PCIe (SM90)**, bf16, torch 2.12 nightly, real config
(40 heads × 128 dim), 2026-05-25.

## What we run today

Self- and cross-attention go through `WanAttention` → the registered custom op
`flash_attn_varlen_func_op` → **FlashAttention‑4** (`flash_attn.cute`):

- `wan/layers/attention/layer.py` — `WanAttention` (calls the op directly; the old
  `USPAttention`/`AttentionBackend`/`FlashAttentionImpl`/metadata layer was removed — it
  was vestigial sequence-parallel scaffolding, dead on single GPU).
- `wan/layers/attention/flash_attn.py` — `flash_attn_varlen_func_op` (+ fake impl for compile).
- `wan/kernels/flash_attention_v4.py` — thin wrapper over `flash_attn.cute`.

Both self-attn (`attn1`) and cross-attn (`attn2`) use `causal=False`,
`softmax_scale = head_dim**-0.5`, dense (no mask). Cross-attention attends over the full
512-token padded text context — this matches the diffusers-convention Wan the checkpoints
were trained for; **do not** add a key mask (it degrades quality).

## Why attention dominates — first-principles cost model

Per self-attention call, the two matmuls (`QKᵀ`, `PV`) cost `4·N²·D·H` FLOPs (softmax is
cheap elementwise). Token count `N = (frames/p_t)·(H/8/p_h)·(W/8/p_w)`:

| res | N (tokens) | self-attn FLOPs/call | note |
|---|--:|--:|---|
| 720p (1280×720×81) | 21·45·80 = **75,600** | `4·N²·128·40` = **117 TFLOP** | dominant |
| 480p (832×480×81)  | 21·30·52 = **32,760** | **22 TFLOP** | ~5.3× cheaper (O(N²)) |

This **O(N²)** is why video is attention-bound: 720p has ~335× the per-head attention work
of a 4K-token LLM. The model predicts the **independent** validation: `117 TFLOP ÷ 412 TFLOP/s
= 284 ms`, which is exactly the measured 720p self-attn latency. The FLOP model is correct.

## Profile breakdown (one 720p step)

From `logs/trace_*.trace.json.gz` (GPU-time share; absolute ms inflated ~1.5× by profiler):

| category | % GPU | detail |
|---|--:|---|
| **attention (FA4)** | **71%** | 40 self-attn (~70%) + 40 cross-attn (~0.7%) |
| gemm (cuBLAS) | 23% | qkv/o projections + FFN |
| elementwise/copy/cast, norm, gelu, rope | ~6% | the fusable "glue" |

**GPU idle ≈ 0%** (timeline-union active ≈ span): the model is **compute-bound and saturated,
not launch-bound.** Implication: `torch.compile`/CUDA-graphs (which recover idle + fuse small
ops) have a **≤6% ceiling** here — attention is opaque to it, GEMMs are already cuBLAS, and
there's no idle to reclaim. Not worth it for this model.

## Backend comparison — latency (720p self, the case that matters)

| backend | latency | TFLOP/s | vs FA4 |
|---|--:|--:|--:|
| **FA4 (cute)** | **284 ms** | **412–416** | **1.00×** |
| FA3 (hopper) | 315–321 ms | 372 | 0.89× |
| SDPA cuDNN | 406 ms | 289–313 | 0.70× |
| SDPA Flash (FA2) | 613 ms | 191 | 0.46× |

- **FA4 wins at 720p self** (the ~70%). FA3 only edges ahead at 480p (~11%) and on tiny
  cross-attn (N=512, launch-bound) — not worth switching for. **Never fall back to SDPA-Flash (FA2).**
- All outputs match FA4 to bf16 rounding (max|Δ| 1e-4–4e-3) → apples-to-apples.
- FA4's 2-CTA kernel needs SM100 (Blackwell) and is **off** on this SM90 card — yet FA4 still
  wins, so a B200 would widen its lead.

## Utilization (MFU) — how close to the ceiling

The datasheet "756 TFLOP/s dense bf16" is unreachable: a **measured** square bf16 GEMM peaks at
**534 TFLOP/s** on this PCIe card (clock/power-limited). So 534 is the honest, softmax-tax-free
ceiling. Against it:

| shape | backend | TFLOP/s | MFU vs GEMM-peak (534) | MFU vs datasheet (756) |
|---|---|--:|--:|--:|
| **720p self** | **FA4** | 416 | **78%** | 55% |
| 720p self | FA3 | 372 | 70% | 49% |
| 720p self | SDPA | 313 | 59% | 41% |
| 480p self | FA4 | 321 | 60% | 42% |

**FA4 runs 720p self-attn at 78% of what pure matmul achieves on this GPU** — i.e. the softmax
tax is only ~22%, and there is essentially no slack left in the kernel. (The 55%-of-datasheet
figure understates it; nothing, not even GEMM, reaches the datasheet number here.) This is the
hard evidence that you can't *tune* attention faster — only reduce its FLOPs.

## What this means for speedups

Since time = FLOPs ÷ throughput and throughput is ~maxed, the only levers reduce FLOPs — all
tradeoffs:

1. **Sparse / windowed attention (VSA)** — exploit video's spatial+temporal locality to skip the
   far-apart token pairs that contribute ~nothing. Directly attacks the 71%; FastVideo ships a
   `WanTransformerBlock_VSA`. Biggest lever; **validate quality**.
2. **Token merging** — collapse redundant (background) tokens before attention → smaller N → N².
3. **Lower resolution** — O(N²): 480p self-attn is ~5.3× cheaper than 720p, by construction.
4. **fp8 attention/GEMM** — ~2× tensor throughput (halves the 23% GEMM too), at a precision cost.

Not worth it here: `torch.compile`/cudagraphs (≤6%, 0% idle), or swapping attention backends
(FA4 already wins and is near-ceiling).

## Reproduce

```bash
uv run python bench_attention.py        # latency + correctness: FA4 / FA3 / SDPA-cuDNN / SDPA-Flash
uv run python bench_attention_util.py   # utilization: achieved TFLOP/s + MFU vs measured GEMM peak
```

Both derive shapes from `WanI2VConfig` (real heads/dim/dtype, 720p/480p, self + cross) and
auto-skip shapes that don't fit if the GPU is busy. `MFU vs GEMM-peak` is the metric to watch —
it stays meaningful across GPUs/clocks in a way the datasheet number doesn't.
