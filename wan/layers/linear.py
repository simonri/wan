from abc import abstractmethod

import torch
import torch.nn as nn

from wan.layers.quantization.config.base_config import QuantizationConfig, QuantizeMethodBase


class LinearBase(torch.nn.Module):
  def __init__(
    self,
    input_size: int,
    output_size: int,
    quant_config: QuantizationConfig,
  ):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_size
    self.quant_config = quant_config
    self.quant_method = quant_config.get_quant_method(self)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class Fp8Linear(LinearBase):
  def __init__(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    quant_config: QuantizationConfig | None = None,
  ):
    super().__init__(input_size, output_size, quant_config)

    self.quant_method.create_weights(self, input_size, output_size)

    if bias:
      self.bias = nn.Parameter(torch.empty(output_size))
    else:
      self.register_parameter("bias", None)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.quant_method.apply(self, x, self.bias)


class LinearMethodBase(QuantizeMethodBase):
  @abstractmethod
  def create_weights(
    self,
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
  ):
    raise NotImplementedError

  @abstractmethod
  def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
  ) -> torch.Tensor:
    raise NotImplementedError
