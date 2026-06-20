import torch
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope_inplace

from wan.utils.custom_op import register_custom_op_from_extern

flashinfer_apply_rope_inplace = register_custom_op_from_extern(
  _flashinfer_apply_rope_inplace,
  op_name="flashinfer_apply_rope_with_cos_sin_cache_inplace",
  mutates_args=["query", "key"],
)


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
