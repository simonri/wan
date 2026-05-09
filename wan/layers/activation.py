import torch
from torch import nn

_ACTIVATION_REGISTRY = {
  "gelu": nn.GELU,
  "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
  "silu": nn.SiLU,
}


def get_act_fn(act_fn_name: str) -> torch.nn.Module:
  act_fn_name = act_fn_name.lower()
  if act_fn_name not in _ACTIVATION_REGISTRY:
    raise ValueError(f"Activation function: {act_fn_name!r} is not supported")

  return _ACTIVATION_REGISTRY[act_fn_name]()
