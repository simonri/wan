import torch
from torch import nn

from wan.layers.activation import get_act_fn
from wan.layers.linear import Fp8Linear
from wan.layers.quantization.config.base_config import QuantizationConfig


class MLP(nn.Module):
  def __init__(
    self,
    input_dim: int,
    mlp_hidden_dim: int,
    output_dim: int | None = None,
    act_type: str = "gelu_pytorch_tanh",
    quant_config: QuantizationConfig | None = None,
  ):
    super().__init__()

    self.fc_in = Fp8Linear(input_dim, mlp_hidden_dim, bias=True, quant_config=quant_config) if quant_config is not None else nn.Linear(input_dim, mlp_hidden_dim, bias=True)

    self.act = get_act_fn(act_type)
    if output_dim is None:
      output_dim = input_dim

    self.fc_out = Fp8Linear(mlp_hidden_dim, output_dim, bias=True, quant_config=quant_config) if quant_config is not None else nn.Linear(mlp_hidden_dim, output_dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.fc_in(x)
    x = self.act(x)
    x = self.fc_out(x)
    return x
