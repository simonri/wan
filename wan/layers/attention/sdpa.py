import torch

from wan.layers.attention.attention_backend import AttentionBackend, AttentionImpl


class SDPABackend(AttentionBackend):
  @staticmethod
  def get_impl_cls() -> type["SDPAImpl"]:
    return SDPAImpl


class SDPAImpl(AttentionImpl):
  def __init__(
    self,
    num_heads: int,
    head_size: int,
    causal: bool,
    softmax_scale: float,
    num_kv_heads: int | None = None,
    prefix: str = "",
    **extra_impl_args,
  ) -> None:
    self.causal = causal
    self.softmax_scale = softmax_scale
    self.dropout = extra_impl_args.get("dropout", 0.0)

  def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_kwargs = {
      "attn_mask": None,
      "dropout_p": self.dropout,
      "is_causal": self.causal,
      "scale": self.softmax_scale,
    }
    if query.shape[1] != key.shape[1]:
      attn_kwargs["enable_gqa"] = True

    output = torch.nn.functional.scaled_dot_product_attention(query, key, value, **attn_kwargs)
    return output.transpose(1, 2)
