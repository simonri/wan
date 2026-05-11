from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
  @staticmethod
  @abstractmethod
  def get_impl_cls() -> "type[AttentionImpl]":
    raise NotImplementedError


class AttentionImpl[T](ABC):
  @abstractmethod
  def __init__(
    self,
    num_heads: int,
    head_size: int,
    softmax_scale: float,
    causal: bool = False,
    num_kv_heads: int | None = None,
    prefix: str = "",
    **extra_impl_args,
  ) -> None:
    raise NotImplementedError

  def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: T) -> torch.Tensor:
    """Preprocess QKV tensor before performing attention operation.

    Default implementation returns the tensor unchanged.
    Subclasses can override this to implement custom preprocessing
    like reshaping, tiling, scaling, or other transformations.

    Called AFTER all_to_all for distributed attention

    """
    return qkv

  def postprocess_output(
    self,
    output: torch.Tensor,
    attn_metadata: T,
  ) -> torch.Tensor:
    """Postprocess the output tensor after the attention operation.

    Default implementation returns the tensor unchanged.
    Subclasses can override this to implement custom postprocessing
    like untiling, scaling, or other transformations.

    Called BEFORE all_to_all for distributed attention

    """

    return output

  @abstractmethod
  def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: T,
  ) -> torch.Tensor:
    raise NotImplementedError
