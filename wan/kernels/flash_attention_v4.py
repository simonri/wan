from __future__ import annotations

from collections.abc import Callable

import torch
from flash_attn.cute import flash_attn_varlen_func as _flash_attn_varlen_func


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
):
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
    return_lse=False,
  )

  if isinstance(result, tuple):
    return result[0]
  return result
