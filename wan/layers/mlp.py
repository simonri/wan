import torch
from torch import nn

from wan.layers.activation import get_act_fn


class MLP(nn.Module):
  def __init__(
    self,
    input_dim: int,
    mlp_hidden_dim: int,
    output_dim: int | None = None,
    act_type: str = "gelu_pytorch_tanh",
  ):
    super().__init__()

    self.fc_in = nn.Linear(input_dim, mlp_hidden_dim, bias=True)

    self.act = get_act_fn(act_type)
    # cuBLASLt's GELU epilogue is the tanh approximation — the same function as
    # gelu_pytorch_tanh — so fc_in + bias + gelu can run as one fused GEMM.
    self._gelu_epilogue_ok = act_type == "gelu_pytorch_tanh"
    if output_dim is None:
      output_dim = input_dim

    self.fc_out = nn.Linear(mlp_hidden_dim, output_dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    fc_in = self.fc_in
    # The epilogue applies gelu inside the GEMM, so it is only valid when no
    # runtime LoRA correction has to be added between fc_in and the activation.
    if (
      self._gelu_epilogue_ok
      and x.is_cuda
      and not getattr(fc_in, "has_runtime_lora", False)
      and x.dtype == fc_in.weight.dtype
    ):
      h = torch._addmm_activation(fc_in.bias, x.reshape(-1, x.shape[-1]), fc_in.weight.t(), use_gelu=True)
      return self.fc_out(h.view(*x.shape[:-1], h.shape[-1]))

    x = fc_in(x)
    x = self.act(x)
    x = self.fc_out(x)
    return x
