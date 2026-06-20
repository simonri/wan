import torch
from torch import nn

from wan.layers.attention.flash_attn import flash_attn_varlen_func_op


class WanAttention(nn.Module):
  def __init__(
    self,
    num_heads: int,
    head_size: int,
    num_kv_heads: int | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    dropout_rate: float = 0.0,
  ) -> None:
    super().__init__()
    self.num_heads = num_heads
    self.head_size = head_size
    self.num_kv_heads = num_kv_heads
    self.softmax_scale = softmax_scale
    self.causal = causal
    self.dropout_p = dropout_rate

  def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return flash_attn_varlen_func_op(
      q=q,
      k=k,
      v=v,
      cu_seqlens_q=None,
      cu_seqlens_k=None,
      max_seqlen_q=q.shape[1],
      max_seqlen_k=k.shape[1],
      softmax_scale=self.softmax_scale,
      causal=self.causal,
    )
