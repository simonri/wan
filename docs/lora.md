# How LoRA works

How LoRA adapters are loaded, applied, swapped, and unloaded in the Wan2.2 I2V
pipeline. Relevant files:

- `wan/layers/lora/linear.py` — `LinearWithLoRA` (the per-layer wrapper + merge/unmerge)
- `wan/pipeline/lora_pipeline.py` — `LoRAPipeline` (`set_lora`, `deactivate_lora_weights`, adapter cache)
- `wan/pipeline/lora_format_adapter.py` — adapter format detection/normalization

## The layer wrapper

Every `nn.Linear` in both transformers is wrapped in a `LinearWithLoRA`, which holds
the original layer as `self.base_layer` plus the state needed to merge and unmerge:
`merged`, `disable_lora`, a `lora_weights_list` of stacked adapters, and a `cpu_weight`
pristine-weight snapshot. Its `forward` is a pass-through — LoRA is **always** baked
into `base_layer.weight`, so there is nothing to compute at call time:

```python
def forward(self, x):
    return self.base_layer(x)   # LoRA already merged into base_layer.weight
```

The wrapper exists so the merged delta can be **unmerged** later (swap/deactivate)
by restoring `cpu_weight`; it adds no per-call math. There is no per-call `torch.compile`
on the layer — the forward is a bare GEMM, so compile (if any) belongs at the
model/block level, not here.

> **Stacking is merge-only.** Multiple adapters on one layer (e.g. lightning +  
> other) are summed into the weight by `_merge_lora_into_data`, which folds every
> entry in `lora_weights_list`. There is no runtime "dynamic" path that applies
> `A`/`B` during the forward — see [Re-adding a dynamic path](#re-adding-a-dynamic-path).

## Lifecycle (what `set_lora` does)

`set_lora(lora_nicknames, lora_paths, targets, strengths)` takes four **parallel
lists** — index `i` applies adapter `i` to target `i` at strength `i`. Targets are
`"transformer"` (high-noise), `"transformer_2"` (low-noise), or `"all"`.

1. **Wrap once.** First call runs `convert_to_lora_layers()` — replaces every
  `nn.Linear` with `LinearWithLoRA` and records `{module_name: layer}` in
   `self.lora_layers` / `self.lora_layers_transformer_2`. ~0.03 s, done once.
2. **Load + cache adapters.** For each nickname not already cached (`load_lora_adapter`):
  read safetensors → `normalize_lora_state_dict` detects the format
   (kohya / wan / diffusers) → rename keys to canonical module names via
   `lora_param_names_mapping` → store on GPU in `self.lora_adapters[nickname]`.
   Cached by nickname, so re-using an adapter is free.
3. **Apply per target** (`_apply_lora_to_layers`, `clear_existing=True`). For each
  layer an adapter covers: infer rank/alpha, then `set_lora_weights(...)` (clears the
   layer's list on the first adapter, then stacks, then merges).
   Layers no adapter covers → `disable_lora = True`.
4. **Merge** (`merge_lora_weights`): unmerge if already merged, snapshot the pristine
  weight to CPU **once** (`_ensure_cpu_weight_snapshot`), then bake
   `Σ (B@A · scale)` into the base weight in place and set `merged = True`.

### Key semantics

- **Replace, not accumulate, across calls.** Each `set_lora` resets the targeted
layers from pristine and re-merges only the adapters you passed. (Re-applying
`lightning` alone cleanly drops a previously-merged `other`, since lightning
covers the same layers.)
- **Target-scoped.** Untargeted transformers keep their current state. A target you
*stop* passing is **not** unloaded — use `deactivate_lora_weights` for that.
- **Always merges.** `set_lora_weights` unconditionally calls `merge_lora_weights`;
there is no `merge_weights` flag to skip it.

## Unloading: `deactivate_lora_weights(target)`

Returns a target to its pristine base:

```python
for layer in target_layers:
    if layer.merged:
        layer.unmerge_lora_weights()   # restore pristine base from the CPU snapshot
    layer.disable_lora = True          # guard: block an accidental re-merge
```

Needed because `set_lora` can only *replace* adapters on layers a new adapter
covers — it can't express "this target back to base." Use this when a target has no
replacement adapter (e.g. dropping `other` from the high-noise transformer,
whose base is already the lightx2v 4-step checkpoint).

> The forward is always `base_layer(x)`, so `unmerge_lora_weights` (which restores the
> pristine weight) is what actually returns the layer to base. Setting
> `disable_lora = True` is a guard: `merge_lora_weights` early-returns when it's set,
> so a stray re-merge can't quietly re-apply the adapter until the next `set_lora`.

## Cost profile (measured, A14B @ bf16)


| operation                      | cost                        | why                                                                                 |
| ------------------------------ | --------------------------- | ----------------------------------------------------------------------------------- |
| convert to LoRA layers         | ~0.03 s (one-time)          | just wraps modules                                                                  |
| adapter load (per file)        | ~0.1–4 s (one-time, cached) | small read + rename                                                                 |
| **first `set_lora*`*           | **~60 s (one-time)**        | CPU snapshot of both transformers (28 GB each, GPU→CPU pageable, ~0.9 GB/s)         |
| subsequent swap / `deactivate` | **~1.4 s / transformer**    | snapshot already cached → only unmerge-restore (CPU→GPU) + re-merge matmul (~0.3 s) |


The expensive `_ensure_cpu_weight_snapshot` runs **once** (guarded by
`cpu_weight is None`); steady-state swaps are cheap. Relative to a ~200 s 720p
denoise, a 1.4 s swap is <1%.

## Why merge-only

The pipeline **always merges**. The runtime "dynamic" path — computing
`(x @ Aᵀ @ Bᵀ) * scale` in the forward instead of baking it into the weight — was
**removed** (the codebase never used it: `set_lora` always merged, so the branch was
dead, and it carried a per-layer `torch.compile` that fragmented the graph). The
merged path is strictly faster for our usage: a bare GEMM at inference (see
`attention-backends.md`), and steady-state swaps between requests cost ~1.4 s/transformer,
negligible vs a ~200 s denoise — so a long-lived swapping server is fine.

### Re-adding a dynamic path

If a use case below appears, a dynamic path would have to be reintroduced — it is not
in the code today:

1. **Mixed adapters in one batch** — different LoRA per concurrent request in the same
  forward pass (merged bakes one set into shared weights; can't). The classic
   multi-LoRA-serving case.
2. **Switching dominates compute** — tiny/few-step inference where the ~1.4 s swap
  isn't negligible, or rapidly probing many adapters.
3. **Frequent strength sweeps** — dynamic changes a scalar; merged needs unmerge+remerge.
4. **Quantized base weights** — can't cleanly bake a bf16 delta into fp8/int8.

Restoring it would mean: keep the `A`/`B` tensors live on each layer, branch in
`forward` on `merged`/`disable_lora`, and **sum the full `lora_weights_list`** (not just
the last adapter) so stacking stays correct. Mind that a per-layer `torch.compile` on
that branch is what we removed for speed — compile at the model/block level instead.