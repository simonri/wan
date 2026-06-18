# FP8 + LoRA: how it works and why

This explains how LoRA adapters are applied to FP8-quantized linear layers in this
codebase, why naïve approaches fail for large-delta LoRAs (like lightning models), and
why we use weight-space dequantization instead of an activation-space bypass.

Relevant files:

- [wan/layers/lora/linear.py](../wan/layers/lora/linear.py) — `Fp8LinearWithLoRA` (per-layer forward)
- [wan/layers/linear.py](../wan/layers/linear.py) — `Fp8Linear` (base FP8 layer)
- [wan/layers/quantization/fp8_utils.py](../wan/layers/quantization/fp8_utils.py) — `apply_fp8_linear`

---

## 1. How FP8 weights are stored

Each `Fp8Linear` layer stores two tensors:

- `weight` — the actual weight matrix in `float8_e4m3fn` dtype, shape `[out, in]`
- `weight_scale` — a single `float32` scalar that records the original magnitude

The relationship is:

```
real_weight = weight * weight_scale
```

`float8_e4m3fn` has only 3 mantissa bits (max value 448.0). To represent large weights
without clipping, the values are divided by `weight_scale` before storage, and
multiplied back when used. A weight with large values gets a large `weight_scale`.

When running inference **without** LoRA, `apply_fp8_linear` does:

```python
# 1. Quantize the input activations to FP8 (per-token scaling)
qinput, x_scale = per_token_quant_fp8(x)

# 2. FP8 × FP8 matmul using CUDA's _scaled_mm
output = torch._scaled_mm(qinput, weight.T,
                           scale_a=x_scale, scale_b=weight_scale,
                           out_dtype=input.dtype)
```

Both weight and input are in FP8 for this matmul. CUDA's `_scaled_mm` kernel uses
FP8 tensor cores, which are ~4× faster than FP16 tensor cores on H100.

---

## 2. What LoRA adds

A LoRA adapter adds a low-rank delta to a weight matrix:

```
effective_weight = W + scale * (B @ A)
```

Where:
- `A` is `[rank, in_features]` — the "down" projection
- `B` is `[out_features, rank]` — the "up" projection
- `scale = strength * (alpha / rank)`
- `rank` is typically 16–64 (very small compared to `in_features` which is 5120)

For most LoRAs (style, character, concept), the delta `B @ A` is small relative to `W`.

**Lightning LoRAs are different.** They are trained to shift the entire denoising
trajectory (multi-step → 4-step), which requires large weight changes. The delta
`B @ A` can be comparable in magnitude to `W` itself.

---

## 3. The broken approach: merge delta into FP8

The most straightforward approach: add the delta directly to the stored FP8 weight.

```python
# (broken)
delta = (B @ A * scale).to(weight.dtype)   # cast delta to float8_e4m3fn
weight.data.add_(delta)                     # add in-place
```

**Why this fails:** FP8 has only 3 mantissa bits. When you add a large delta to an FP8
weight and re-quantize back to FP8, you permanently lose precision in the result.
`float8_e4m3fn` can represent values like `0.5, 0.5078125, 0.515625...` — the gaps
between representable values at large magnitudes are huge. For lightning LoRAs the
delta is large, so the quantization error after baking it in is severe.

This is what caused bad results with lightning LoRAs. The LoRA was loaded and appeared
to apply, but the FP8 re-quantization destroyed the precision of the merged weight.

---

## 4. The tempting fix: activation-space bypass

The obvious alternative: keep the FP8 weight untouched and add the LoRA correction
**after** the FP8 matmul, in the output space.

```python
# (discarded — see why below)
out = fp8_matmul(x, W)                          # FP8 × FP8 base matmul
out += F.linear(F.linear(x, A), scaled_B)       # LoRA correction in fp16
```

Mathematically this looks equivalent to `(W + B@A) @ x`, but it is **not**.

### Why activation-space has precision loss

The `fp8_matmul` call internally quantizes the **input** `x` to FP8 before the matmul:

```python
# inside apply_fp8_linear:
qinput, x_scale = per_token_quant_fp8(x)   # x → float8_e4m3fn
output = _scaled_mm(qinput, W_fp8, ...)     # uses FP8 x
```

The LoRA correction `F.linear(x, A)` uses the **original** full-precision `x`.

So the two branches operate on different versions of x:

```
activation-space result = W @ x_fp8 + B@A @ x_fp16
```

These are not the same x. The true answer is `(W + B@A) @ x_fp16`. The error is:

```
error = W @ (x_fp8 - x_fp16)
```

This error term scales with `W` (the full weight matrix), not with the delta. For
large base weights (which FP8 models have, since `weight_scale` can be large), the
error from quantizing the input to FP8 is significant and inconsistently applied
between base and LoRA paths.

For standard small-delta LoRAs this error may be tolerable. For lightning LoRAs
where precision matters (the delta is doing heavy lifting), the inconsistency
produces visibly bad results.

---

## 5. The correct approach: weight-space dequantization

Dequantize the FP8 weight to fp16 **before** the matmul, apply the LoRA delta in
weight-space, then do a single fp16 matmul.

```python
# Step 1: dequantize fp8 weight to activation dtype (fp16 or bf16)
#         weight_scale is a Python float so no extra tensor is created
weight = base_layer.weight.data.to(x.dtype) * base_layer.weight_scale.item()

# Step 2: apply LoRA delta(s) in-place in weight-space
for lora_A, scaled_lora_B in lora_cache:
    weight.add_(scaled_lora_B @ lora_A)

# Step 3: single fp16 matmul — x is never quantized to FP8
return F.linear(x, weight, bias)
```

This computes `(W_fp16 + B@A) @ x_fp16` exactly. The input `x` is **never** quantized
to FP8. The only approximation is the weight dequantization step (fp8 → fp16), which
introduces a small and uniform error — not the large inconsistent error from
quantizing the input.

### Why this is what ComfyUI does

ComfyUI's `ops.py` (the `forward_comfy_cast_weights` method) does the same thing:
when a LoRA `weight_function` is attached, it skips `fp8_linear()` entirely and falls
through to `cast_bias_weight(weight, input)` + `F.linear(input, weight, bias)`. Their
`cast_bias_weight` dequantizes the fp8 weight to the input's dtype before the matmul.

---

## 6. The performance tradeoff

Weight-space LoRA gives up the FP8 tensor core speedup for LoRA-active layers:

| Path | Matmul dtype | Speed |
|------|-------------|-------|
| No LoRA | FP8 × FP8 | ~4× vs fp16 |
| Weight-space LoRA | fp16 × fp16 | baseline |

For LoRA-inactive layers (or when LoRA is deactivated), `forward` falls back to
`self.base_layer(x)` which uses the full FP8 path.

### Why keep FP8 weights if we dequantize for LoRA?

Two reasons:

1. **Memory.** FP8 weights use half the VRAM of fp16 (14 GB vs 28 GB for 14B params).
   This is the primary reason the model fits on a single GPU.
2. **Reading from HBM is cheaper.** Even though we dequantize before the matmul, we
   read 26 MB fp8 from HBM and write 52 MB fp16. If weights were stored in fp16 we'd
   read 52 MB directly. The fp8 read is 2× cheaper, partially offsetting the dequant
   overhead.

---

## 7. The `_lora_cache`

To avoid repeating the dtype casts and scale folding every forward step, the layer
builds a cache on the first forward call after activation:

```python
if self._lora_cache is None:
    cache = []
    for lora_A, lora_B, _, strength, rank, alpha in self.lora_weights_list:
        scale = strength
        if alpha is not None and rank is not None and alpha != rank:
            scale *= alpha / rank
        cache.append((lora_A.to(x.dtype), (lora_B * scale).to(x.dtype)))
    self._lora_cache = cache
```

- Scale is folded into `B` once: `scaled_B = B * (strength * alpha/rank)`
- Both `A` and `scaled_B` are cast to the activation dtype once
- Cache is invalidated (`= None`) by `merge_lora_weights` / `unmerge_lora_weights`

This means per-step cost is: one fp8→fp16 dequant + one or more `weight.add_` +
one `F.linear`. No per-step tensor allocations for scale or dtype conversion.

---

## 8. The "merge" terminology

`merge_lora_weights` and `unmerge_lora_weights` are named after the traditional
merge-based LoRA approach but do **not** modify the FP8 weight here. They only
set the `merged` flag and clear the cache:

- `merged = True` → `forward` uses the weight-space dequant path
- `merged = False` / `disable_lora = True` → `forward` returns `self.base_layer(x)` (pure FP8)

The FP8 base weight is never modified. Deactivating LoRA is instant (no
weight restoration needed), unlike the legacy merge-based approach which required
a CPU snapshot and memcpy.

---

## Summary

| Approach | What it does | Problem |
|----------|-------------|---------|
| Merge delta into FP8 | `W_fp8 += delta.to(fp8)` | Re-quantization destroys precision for large deltas |
| Activation-space bypass | `W_fp8 @ x_fp8 + B@A @ x_fp16` | Input quantized inconsistently: base uses `x_fp8`, LoRA uses `x_fp16` |
| **Weight-space dequant (chosen)** | `(W_fp16 + B@A) @ x_fp16` | Only the weight has fp8 error; input always full precision |
