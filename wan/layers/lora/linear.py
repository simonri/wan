import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.layers.linear import Fp8Linear

LoRAWeightEntry = tuple[
  torch.nn.Parameter,
  torch.nn.Parameter,
  str | None,
  float,
  int | None,
  int | None,
]


class BaseLayerWithLoRA(nn.Module):
  def __init__(
    self,
    base_layer: nn.Module,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
  ):
    super().__init__()
    self.base_layer: nn.Module = base_layer
    self.merged: bool = False
    self.disable_lora: bool = True
    self.lora_rank = lora_rank
    self.lora_alpha = lora_alpha
    self.lora_weights_list: list[LoRAWeightEntry] = []

  @property
  def weight(self) -> torch.Tensor:
    return self.base_layer.weight

  @property
  def bias(self) -> torch.Tensor:
    return self.base_layer.bias

  @torch.no_grad()
  def merge_lora_weights(self) -> None:
    raise NotImplementedError

  @torch.no_grad()
  def unmerge_lora_weights(self) -> None:
    raise NotImplementedError

  def set_lora_weights(
    self,
    A: torch.Tensor,
    B: torch.Tensor,
    lora_path: str | None = None,
    strength: float = 1.0,
    clear_existing: bool = False,
  ) -> None:
    lora_A_param = torch.nn.Parameter(A)
    lora_B_param = torch.nn.Parameter(B)

    if clear_existing:
      self.lora_weights_list.clear()

    self.lora_weights_list.append(
      (
        lora_A_param,
        lora_B_param,
        lora_path,
        strength,
        self.lora_rank,
        self.lora_alpha,
      )
    )

    self.disable_lora = False
    self.merge_lora_weights()


class Fp8LinearWithLoRA(BaseLayerWithLoRA):
  def __init__(
    self,
    base_layer: Fp8Linear,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
  ) -> None:
    super().__init__(base_layer, lora_rank, lora_alpha)
    # Lazily built on first forward after activation: (A, scaled_B) pairs cast to the
    # activation dtype. Avoids per-step allocations and the [M, out_features] scale multiply.
    self._lora_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

  @torch.no_grad()
  def merge_lora_weights(self) -> None:
    if self.disable_lora:
      return
    if self.merged:
      self.unmerge_lora_weights()
    if not self.lora_weights_list:
      raise ValueError("LoRA weights not set. Please set them first.")
    # Base FP8 weight is never modified — LoRA is applied as a float bypass in forward().
    self._lora_cache = None
    self.merged = True

  @torch.no_grad()
  def unmerge_lora_weights(self) -> None:
    if self.disable_lora:
      return
    if not self.merged:
      raise ValueError("LoRA weights are not merged. Please merge them first.")
    # Nothing to restore — base FP8 weight was never modified.
    self._lora_cache = None
    self.merged = False

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.disable_lora or not self.merged or not self.lora_weights_list:
      return self.base_layer(x)

    if self._lora_cache is None:
      # Build once per activation: cast to activation dtype and fold scale into B.
      cache = []
      for lora_A, lora_B, _, lora_strength, lora_rank, lora_alpha in self.lora_weights_list:
        scale = lora_strength
        if lora_alpha is not None and lora_rank is not None and lora_alpha != lora_rank:
          scale *= lora_alpha / lora_rank
        cache.append((lora_A.to(dtype=x.dtype), (lora_B * scale).to(dtype=x.dtype)))
      self._lora_cache = cache

    # Dequantize fp8 weight to activation dtype. weight_scale is a per-tensor scalar.
    weight = self.base_layer.weight.data.to(x.dtype) * self.base_layer.weight_scale.item()

    # Apply LoRA in weight-space: single fp16 matmul over the combined weight.
    # Avoids fp8 input quantization error — see docs/fp8-lora.md.
    for lora_A, scaled_lora_B in self._lora_cache:
      weight.add_(scaled_lora_B @ lora_A)

    return F.linear(x, weight, self.base_layer.bias)


def wrap_with_lora_layer(
  layer: nn.Module,
  lora_rank: int | None = None,
  lora_alpha: int | None = None,
) -> nn.Module | None:
  """
  Transform the given layer to its corresponding LoRA layer.
  """
  supported_layer_types: dict = {
    Fp8Linear: Fp8LinearWithLoRA,
  }

  for src_layer_type, lora_layer_type in supported_layer_types.items():
    if isinstance(layer, src_layer_type):
      ret = lora_layer_type(
        layer,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
      )
      return ret
  return None


def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> nn.Module:
  """Replace a submodule in a model with a new module."""
  parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
  target_name = module_name.split(".")[-1]
  setattr(parent, target_name, new_module)
  return new_module
