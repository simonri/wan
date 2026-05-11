import torch
from torch import nn

from wan.layers.attention.attention_backend import AttentionImpl
from wan.layers.attention.flash_attn import FlashAttentionBackend


class USPAttention(nn.Module):
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

    dtype = torch.get_default_dtype()
    attn_backend = FlashAttentionBackend()

    impl_cls: type[AttentionImpl] = attn_backend.get_impl_cls()
    self.attn_impl = impl_cls(
      num_heads=num_heads,
      head_size=head_size,
      causal=causal,
      softmax_scale=softmax_scale,
      num_kv_heads=num_kv_heads,
    )
    self.num_heads = num_heads
    self.head_size = head_size
    self.num_kv_heads = num_kv_heads
    self.dtype = dtype
    self.causal = causal
    self.dropout_p = dropout_rate

  def forward(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    ctx_attn_metadata = None
    out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
    return out
