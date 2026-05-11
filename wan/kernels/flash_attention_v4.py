from __future__ import annotations

from collections.abc import Callable

import torch

try:
  from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func
except Exception as _e:  # pragma: no cover
  _flash_attn_varlen_func = None
  _flash_attn_import_error = _e
else:
  _flash_attn_import_error = None


def _maybe_contiguous(x: torch.Tensor | None) -> torch.Tensor | None:
  return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_varlen_func(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k: torch.Tensor | None = None,
  seqused_q: torch.Tensor | None = None,
  seqused_k: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  max_seqlen_k: int | None = None,
  page_table: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  softcap: float | None = None,
  window_size: tuple[int | None, int | None] = (-1, -1),
  learnable_sink: torch.Tensor | None = None,
  sinks: torch.Tensor | None = None,
  num_splits: int = 1,
  pack_gqa: bool | None = None,
  score_mod: Callable | None = None,
  aux_tensors: list | None = None,
  return_softmax_lse: bool = False,
):
  if _flash_attn_varlen_func is None:  # pragma: no cover
    raise ImportError(
      "Vendored FlashAttention CUTE is not available (cannot import flash_attn.cute). Please check your source tree."
    ) from _flash_attn_import_error

  q, k, v = [_maybe_contiguous(t) for t in (q, k, v)]
  cu_seqlens_q, cu_seqlens_k = [_maybe_contiguous(t) for t in (cu_seqlens_q, cu_seqlens_k)]
  seqused_q, seqused_k = [_maybe_contiguous(t) for t in (seqused_q, seqused_k)]
  page_table = _maybe_contiguous(page_table)

  if learnable_sink is None and sinks is not None:
    learnable_sink = sinks

  if window_size == (-1, -1):
    window_size = (None, None)

  result = _flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    seqused_q=seqused_q,
    seqused_k=seqused_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    page_table=page_table,
    softmax_scale=softmax_scale,
    causal=causal,
    softcap=softcap,
    window_size=window_size,
    learnable_sink=learnable_sink,
    num_splits=num_splits,
    pack_gqa=pack_gqa,
    score_mod=score_mod,
    aux_tensors=aux_tensors,
    return_lse=return_softmax_lse,
  )

  if return_softmax_lse:
    return result
  if isinstance(result, tuple):
    return result[0]
  return result


def flash_attn_with_kvcache(
  q: torch.Tensor,
  k_cache: torch.Tensor,
  v_cache: torch.Tensor,
  k: torch.Tensor | None = None,
  v: torch.Tensor | None = None,
  qv: torch.Tensor | None = None,
  rotary_cos: torch.Tensor | None = None,
  rotary_sin: torch.Tensor | None = None,
  cache_seqlens: int | torch.Tensor | None = None,
  cache_batch_idx: torch.Tensor | None = None,
  cache_leftpad: torch.Tensor | None = None,
  page_table: torch.Tensor | None = None,
  cu_seqlens_q: torch.Tensor | None = None,
  cu_seqlens_k_new: torch.Tensor | None = None,
  max_seqlen_q: int | None = None,
  rotary_seqlens: torch.Tensor | None = None,
  q_descale: torch.Tensor | None = None,
  k_descale: torch.Tensor | None = None,
  v_descale: torch.Tensor | None = None,
  softmax_scale: float | None = None,
  causal: bool = False,
  window_size: tuple[int | None, int | None] = (-1, -1),
  attention_chunk: int | None = None,
  softcap: float = 0.0,
  rotary_interleaved: bool = True,
  scheduler_metadata=None,
  num_splits: int = 0,
  pack_gqa: bool | None = None,
  sm_margin: int = 0,
  sinks: torch.Tensor | None = None,
  score_mod: Callable | None = None,
  aux_tensors: list | None = None,
  return_softmax_lse: bool = False,
  **_: object,
):
  if k is not None or v is not None or qv is not None:
    raise NotImplementedError("FA4 does not support updating KV cache in-place.")
  if rotary_cos is not None or rotary_sin is not None or rotary_seqlens is not None:
    raise NotImplementedError("FA4 path does not support rotary embedding.")
  if cache_batch_idx is not None or cache_leftpad is not None:
    raise NotImplementedError("FA4 path does not support non-consecutive batch indices or left padding.")
  if q_descale is not None or k_descale is not None or v_descale is not None:
    raise NotImplementedError("FA4 path does not support descale.")

  if isinstance(cache_seqlens, int):
    cache_seqlens = torch.full((k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device)

  result = flash_attn_varlen_func(
    q=q,
    k=k_cache,
    v=v_cache,
    cu_seqlens_q=cu_seqlens_q,
    seqused_k=cache_seqlens,
    max_seqlen_q=max_seqlen_q,
    page_table=page_table,
    softmax_scale=softmax_scale,
    causal=causal,
    softcap=softcap if softcap != 0.0 else None,
    window_size=window_size,
    num_splits=num_splits if num_splits != 0 else 1,
    pack_gqa=pack_gqa,
    learnable_sink=sinks,
    score_mod=score_mod,
    aux_tensors=aux_tensors,
    return_softmax_lse=True,
  )

  if return_softmax_lse:
    return result
  if isinstance(result, tuple):
    return result[0]
  return result
