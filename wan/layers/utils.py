from typing import Any

import torch


def set_weight_attrs(weight: torch.Tensor, weight_attrs: dict[str, Any] | None):
  if weight_attrs is None:
    return
  for key, value in weight_attrs.items():
    assert not hasattr(weight, key), f"Overwriting existing tensor attribute {key}"
    setattr(weight, key, value)
