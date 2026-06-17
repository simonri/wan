import torch

from wan.layers.linear import LinearMethodBase
from wan.layers.quantization.config.base_config import QuantizationConfig
from wan.layers.quantization.fp8_utils import apply_fp8_linear


class Fp8Config(QuantizationConfig):
  def __init__(
    self,
    packed_modules_mapping: dict[str, list[str]] | None = None,
  ):
    self.packed_modules_mapping = packed_modules_mapping or {}

  def get_quant_method(self, layer: torch.nn.Module):
    from wan.layers.linear import LinearBase

    if isinstance(layer, LinearBase):
      return Fp8LinearMethod(self)
    return None


class Fp8LinearMethod(LinearMethodBase):
  def __init__(self, quant_config: Fp8Config):
    self.quant_config = quant_config

  def create_weights(
    self,
    layer: torch.nn.Module,
    input_size: int,
    output_size: int,
  ):
    weight = torch.nn.Parameter(torch.empty((output_size, input_size), dtype=torch.float8_e4m3fn))
    layer.register_parameter("weight", weight)

    # per-tensor scale — name matches checkpoint key after param_names_mapping
    scale = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32))
    layer.register_parameter("weight_scale", scale)

  def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return apply_fp8_linear(
      input=x,
      weight=layer.weight,
      weight_scale=layer.weight_scale,
      bias=bias,
    )
