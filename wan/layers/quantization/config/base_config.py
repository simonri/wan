from abc import ABC, abstractmethod

import torch


class QuantizeMethodBase(ABC):
  @abstractmethod
  def create_weights(self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs):
    raise NotImplementedError

  @abstractmethod
  def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
    raise NotImplementedError


class QuantizationConfig(ABC):
  def __init__(self):
    super().__init__()
    self.packed_modules_mapping: dict[str, list[str]] = dict()

  @abstractmethod
  def get_quant_method(self, layer: torch.nn.Module) -> QuantizeMethodBase | None:
    raise NotImplementedError
