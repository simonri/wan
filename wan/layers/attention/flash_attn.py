import torch

from wan.kernels.flash_attention import flash_attn_varlen_func
from wan.utils.custom_op import register_custom_op


def maybe_contiguous(x: torch.Tensor | None) -> torch.Tensor | None:
  return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func_fake_out(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  max_seqlen_k: int | None = None,
  seqused_q: torch.Tensor | None = None,
  seqused_k: torch.Tensor | None = None,
  page_table: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  qv: torch.Tensor | None = None,
  q_descale: torch.Tensor | None = None,
  k_descale: torch.Tensor | None = None,
  v_descale: torch.Tensor | None = None,
  window_size: list[int] | None = None,
  attention_chunk: int = 0,
  softcap: float = 0.0,
  num_splits: int = 1,
  pack_gqa: bool | None = None,
  sm_margin: int = 0,
  return_softmax_lse: bool = False,
  sinks: torch.Tensor | None = None,
) -> torch.Tensor:
  q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
  num_head, head_dim = q.shape[-2:]
  if cu_seqlens_q is None:
    batch_size, seqlen_q = q.shape[:2]
  else:
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = None
  head_dim_v = v.shape[-1]

  if cu_seqlens_q is not None:
    assert cu_seqlens_q.shape == (batch_size + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
    assert cu_seqlens_q.stride(0) == 1, "cu_seqlens_q must be contiguous"

  assert q.dtype in [
    torch.float16,
    torch.bfloat16,
  ], "inputs must be float16 or bfloat16"
  assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
  assert head_dim <= 256, "head_dim must be less than or equal to 256"
  alignment = 16 // q.element_size()
  assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"

  q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (q.shape[0],)
  out = q.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)
  return out


def flash_attn_varlen_func_fake_out_lse(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  max_seqlen_k: int | None = None,
  seqused_q: torch.Tensor | None = None,
  seqused_k: torch.Tensor | None = None,
  page_table: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  qv: torch.Tensor | None = None,
  q_descale: torch.Tensor | None = None,
  k_descale: torch.Tensor | None = None,
  v_descale: torch.Tensor | None = None,
  window_size: list[int] | None = None,
  attention_chunk: int = 0,
  softcap: float = 0.0,
  num_splits: int = 1,
  pack_gqa: bool | None = None,
  sm_margin: int = 0,
  return_softmax_lse: bool = True,
  sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
  num_head, head_dim = q.shape[-2:]
  if cu_seqlens_q is None:
    batch_size, seqlen_q = q.shape[:2]
    total_q = batch_size * seqlen_q
  else:
    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = None
    total_q = q.shape[0]
  head_dim_v = v.shape[-1]

  if cu_seqlens_q is not None:
    assert cu_seqlens_q.shape == (batch_size + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert cu_seqlens_q.dtype == torch.int32, "cu_seqlens_q must be int32"
    assert cu_seqlens_q.stride(0) == 1, "cu_seqlens_q must be contiguous"

  assert q.dtype in [
    torch.float16,
    torch.bfloat16,
  ], "inputs must be float16 or bfloat16"
  assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
  assert head_dim <= 256, "head_dim must be less than or equal to 256"
  alignment = 16 // q.element_size()
  assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"

  q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
  lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)

  out = q.new_empty(*q_batch_seqlen_shape, num_head, head_dim_v)
  lse = q.new_empty(lse_shape, dtype=torch.float32)
  return out, lse


@register_custom_op(fake_impl=flash_attn_varlen_func_fake_out)
def flash_attn_varlen_func_op(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  max_seqlen_k: int | None = None,
  seqused_q: torch.Tensor | None = None,
  seqused_k: torch.Tensor | None = None,
  page_table: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  qv: torch.Tensor | None = None,
  q_descale: torch.Tensor | None = None,
  k_descale: torch.Tensor | None = None,
  v_descale: torch.Tensor | None = None,
  window_size: list[int] | None = None,
  attention_chunk: int = 0,
  softcap: float = 0.0,
  num_splits: int = 1,
  pack_gqa: bool | None = None,
  sm_margin: int = 0,
  return_softmax_lse: bool = False,
  sinks: torch.Tensor | None = None,
) -> torch.Tensor:
  if window_size is None:
    window_size = [-1, -1]
  if return_softmax_lse:
    raise ValueError(
      "flash_attn_varlen_func_op is out-only op; return_softmax_lse must be False. "
      "Use flash_attn_varlen_func_op_lse for (out, lse)."
    )
  return flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    page_table=page_table,
    softmax_scale=softmax_scale,
    causal=causal,
    qv=qv,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
    window_size=tuple(window_size),
    attention_chunk=attention_chunk,
    softcap=softcap,
    num_splits=num_splits,
    pack_gqa=pack_gqa,
    sm_margin=sm_margin,
    return_softmax_lse=False,
    sinks=sinks,
  )


@register_custom_op(fake_impl=flash_attn_varlen_func_fake_out_lse)
def flash_attn_varlen_func_op_lse(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  max_seqlen_k: int | None = None,
  seqused_q: torch.Tensor | None = None,
  seqused_k: torch.Tensor | None = None,
  page_table: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  qv: torch.Tensor | None = None,
  q_descale: torch.Tensor | None = None,
  k_descale: torch.Tensor | None = None,
  v_descale: torch.Tensor | None = None,
  window_size: list[int] | None = None,
  attention_chunk: int = 0,
  softcap: float = 0.0,
  num_splits: int = 1,
  pack_gqa: bool | None = None,
  sm_margin: int = 0,
  return_softmax_lse: bool = True,
  sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if window_size is None:
    window_size = [-1, -1]
  if not return_softmax_lse:
    raise ValueError(
      "flash_attn_varlen_func_op_lse is out+lse op; return_softmax_lse must be True. "
      "Use flash_attn_varlen_func_op for out-only."
    )
  return flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    page_table=page_table,
    softmax_scale=softmax_scale,
    causal=causal,
    qv=qv,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
    window_size=tuple(window_size),
    attention_chunk=attention_chunk,
    softcap=softcap,
    num_splits=num_splits,
    pack_gqa=pack_gqa,
    sm_margin=sm_margin,
    return_softmax_lse=True,
    sinks=sinks,
  )


