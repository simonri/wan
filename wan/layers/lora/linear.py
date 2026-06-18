import torch
import torch.nn as nn
import torch.nn.functional as F

LoRAWeightEntry = tuple[
  torch.nn.Parameter,
  torch.nn.Parameter,
  str | None,
  float,
  int | None,
  int | None,
]

LORA_TARGET_NAMES = {"to_q", "to_k", "to_v", "to_out", "fc_in", "fc_out"}
LORA_IGNORE_PREFIXES = {"condition_embedder"}


class LinearWithLoRA(nn.Module):
  def __init__(self, base_layer: nn.Linear):
    super().__init__()
    self.base_layer = base_layer
    self.disable_lora: bool = True
    self.lora_weights_list: list[LoRAWeightEntry] = []
    self._lora_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

  @property
  def weight(self) -> torch.Tensor:
    return self.base_layer.weight

  @property
  def bias(self) -> torch.Tensor:
    return self.base_layer.bias

  def set_lora_weights(
    self,
    A: torch.Tensor,
    B: torch.Tensor,
    lora_path: str | None = None,
    strength: float = 1.0,
    clear_existing: bool = False,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
  ) -> None:
    if clear_existing:
      self.lora_weights_list.clear()

    self.lora_weights_list.append(
      (
        torch.nn.Parameter(A),
        torch.nn.Parameter(B),
        lora_path,
        strength,
        lora_rank,
        lora_alpha,
      )
    )
    self.disable_lora = False
    self._lora_cache = None

  def deactivate(self) -> None:
    self.disable_lora = True
    self._lora_cache = None

  def _build_cache(self, dtype: torch.dtype) -> None:
    cache = []
    for lora_A, lora_B, _, lora_strength, lora_rank, lora_alpha in self.lora_weights_list:
      scale = lora_strength
      if lora_alpha is not None and lora_rank is not None and lora_alpha != lora_rank:
        scale *= lora_alpha / lora_rank
      cache.append((lora_A.to(dtype=dtype), (lora_B * scale).to(dtype=dtype)))
    self._lora_cache = cache

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = self.base_layer(x)

    if self.disable_lora or not self.lora_weights_list:
      return result

    if self._lora_cache is None:
      self._build_cache(x.dtype)

    for lora_A, scaled_lora_B in self._lora_cache:
      result = result + F.linear(F.linear(x, lora_A), scaled_lora_B)

    return result


def wrap_with_lora_layer(layer: nn.Module, layer_name: str) -> nn.Module | None:
  parts = layer_name.split(".")
  leaf_name = parts[-1]
  if leaf_name not in LORA_TARGET_NAMES:
    return None
  if parts[0] in LORA_IGNORE_PREFIXES:
    return None
  if not isinstance(layer, nn.Linear):
    return None
  return LinearWithLoRA(layer)


def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> nn.Module:
  parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
  target_name = module_name.split(".")[-1]
  setattr(parent, target_name, new_module)
  return new_module
