# Attention backends: FA4 vs FA3 vs SDPA

How the Wan2.2 I2V DiT does attention, and which kernel is actually fastest on our
hardware. **TL;DR: keep FlashAttention-4 (the `flash_attn.cute` path) — it's the
fastest for our dominant shape (720p self-attention) on this H100.**

## What we run today

Self- and cross-attention in the DiT go through `USPAttention` →
`FlashAttentionImpl.forward` → the registered custom op `flash_attn_varlen_func_op`
→ **FlashAttention-4** (`flash_attn.cute`).

- `wan/layers/attention/layer.py` — `USPAttention`
- `wan/layers/attention/flash_attn.py` — `FlashAttentionImpl`, the custom-op wrappers
- `wan/kernels/flash_attention_v4.py` — thin wrapper over `flash_attn.cute`

Both self-attn (`attn1`) and cross-attn (`attn2`) use it with `causal=False`,
`softmax_scale = head_dim**-0.5`, no mask (cross-attention attends over the full
512-token padded text context — this matches the diffusers-convention Wan the
checkpoints were trained for; do **not** add a key mask).

## Benchmark (2026-05-25)

`uv run python bench_attention.py` — forward-only, **bf16**, real config
(40 heads × 128 dim) and real token counts. NVIDIA **H100 PCIe**, torch 2.12
nightly, FA3 (`flash_attn_interface`) and FA4 (`flash_attn.cute`) both installed.

Token counts: 720p (1280×720×81) → **75,600** tokens; 480p (832×480×81) →
**32,760**; text/cross key length 512.

### 720p self-attention — the dominant cost (~95% of attention time)

| backend          | latency | TFLOP/s | vs FA4 |
|------------------|--------:|--------:|-------:|
| **FA4 (cute)**   | **284 ms** | **412** | **1.00×** |
| FA3 (hopper)     | 321 ms  | 365     | 0.89×  |
| SDPA cuDNN       | 406 ms  | 289     | 0.70×  |
| SDPA Flash (FA2) | 613 ms  | 191     | 0.46×  |

### 480p self-attention

| backend          | latency | vs FA4 |
|------------------|--------:|-------:|
| FA3 (hopper)     | 71 ms   | 1.11×  |
| **FA4 (cute)**   | 79 ms   | 1.00×  |
| SDPA cuDNN       | 102 ms  | 0.78×  |
| SDPA Flash (FA2) | 117 ms  | 0.68×  |

### Cross-attention (Nkv = 512)

~1–3 ms regardless of backend — negligible. SDPA-cuDNN is nominally fastest here,
then FA3, then FA4, but the absolute difference is sub-millisecond.

Correctness: every backend's output matched FA4 to within bf16 rounding
(max|Δ| 1e-4 – 4e-3), so the comparison is apples-to-apples.

## Verdict

- **FA4 wins the case that matters.** At 720p self-attention it's the fastest
  (412 TFLOP/s), ahead of FA3, cuDNN, and FA2. Keep it.
- **FA3 is only ahead at 480p (~11%)** and loses at 720p (~11%) — not worth
  switching the attention path for a resolution-specific edge on one component.
- **Never fall back to plain SDPA-Flash (FA2)** — slowest by far (0.46× at 720p).
- Cross-attention backend choice is in the noise.

## Hardware notes

- This is an **H100 PCIe (SM90)**. FA4's headline feature, the **2-CTA kernel**,
  requires **SM100 (Blackwell)** and is disabled here — yet FA4's Hopper path is
  still fastest, so a B200 would likely widen FA4's lead, not flip it.
- At 300–410 TFLOP/s on these shapes, all flash-style kernels are already near
  what this PCIe card delivers. The bigger remaining lever is the **un-compiled
  DiT around the attention** (norms/modulation/residual/FFN run eager), not the
  attention kernel itself.

## Reproduce

```bash
uv run python bench_attention.py
```

`bench_attention.py` derives shapes from `WanI2VConfig` and benches FA4 / FA3 /
SDPA-cuDNN / SDPA-Flash. It auto-skips any shape that won't fit if the GPU is busy,
and prints per-shape latency, TFLOP/s, and correctness Δ vs FA4.
