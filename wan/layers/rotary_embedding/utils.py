import torch

from wan.utils.custom_op import register_custom_op_from_extern

try:
  from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope_inplace
except Exception:
  _flashinfer_apply_rope_inplace = None

if _flashinfer_apply_rope_inplace is not None:
  flashinfer_apply_rope_inplace = register_custom_op_from_extern(
    _flashinfer_apply_rope_inplace,
    op_name="flashinfer_apply_rope_with_cos_sin_cache_inplace",
    mutates_args=["query", "key"],
  )
else:
  flashinfer_apply_rope_inplace = None


def apply_flashinfer_rope_qk_inplace(
  q: torch.Tensor,
  k: torch.Tensor,
  cos_sin_cache: torch.Tensor,
  *,
  head_size: int | None = None,
  is_neox: bool = False,
  positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if q.dim() != 4 or k.dim() != 4:
    raise ValueError()

  if q.shape != k.shape:
    raise ValueError()

  bsz, seqlen, nheads, d = q.shape
  if head_size is None:
    head_size = d
  if head_size != d:
    raise ValueError(f"head_size mismatch: inferred {d}, but head_size={head_size}")

  if flashinfer_apply_rope_inplace is None:
    # Triton fallback for AMD/ROCm where FlashInfer is not available

    _warn_about_missing_flashinfer()

    half_size = cos_sin_cache.shape[-1] // 2
    if positions is None:
      cos = cos_sin_cache[:seqlen, :half_size].to(q.dtype)
      sin = cos_sin_cache[:seqlen, half_size:].to(q.dtype)
      cos = cos.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
      sin = sin.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * seqlen, -1)
    else:
      positions = positions.to(cos_sin_cache.device).view(-1)
      cos = cos_sin_cache[positions, :half_size].to(q.dtype)
      sin = cos_sin_cache[positions, half_size:].to(q.dtype)
    q_flat = q.reshape(bsz * seqlen, nheads, d)
    k_flat = k.reshape(bsz * seqlen, nheads, d)
    q_rot = apply_rotary_embedding(q_flat, cos, sin, interleaved=not is_neox)
    k_rot = apply_rotary_embedding(k_flat, cos, sin, interleaved=not is_neox)
    return q_rot.view(bsz, seqlen, nheads, d), k_rot.view(bsz, seqlen, nheads, d)

  if positions is None:
    pos_1d = torch.arange(seqlen, device=q.device, dtype=torch.long)
    positions = pos_1d if bsz == 1 else pos_1d.repeat(bsz)
  else:
    if not (isinstance(positions, torch.Tensor) and positions.dtype == torch.long and positions.dim() == 1):
      raise ValueError("positions must be a 1D torch.long Tensor")
    if positions.numel() != bsz * seqlen:
      raise ValueError(f"positions length must be bsz*seqlen={bsz * seqlen}, got {positions.numel()}")

  q_flat = q.reshape(bsz * seqlen, nheads * d).contiguous()
  k_flat = k.reshape(bsz * seqlen, nheads * d).contiguous()
  flashinfer_apply_rope_inplace(
    positions=positions,
    query=q_flat,
    key=k_flat,
    head_size=d,
    cos_sin_cache=cos_sin_cache,
    is_neox=is_neox,
  )
  return q_flat.view(bsz, seqlen, nheads, d), k_flat.view(bsz, seqlen, nheads, d)
