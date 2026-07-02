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
    self._disable_lora: bool = True
    self.lora_weights_list: list[LoRAWeightEntry] = []
    self._lora_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    # bumped whenever the effective weights change (adapters set/cleared/toggled)
    # so downstream activation caches keyed on this layer can invalidate
    self.weights_version: int = 0
    # exact CPU copy of the base weight, taken before any merge; restore is a
    # bitwise copy back, so merge/restore cycles never accumulate rounding drift
    self._pristine_weight: torch.Tensor | None = None
    self.is_merged: bool = False

  @property
  def has_runtime_lora(self) -> bool:
    return bool(self.lora_weights_list) and not self._disable_lora

  def snapshot_pristine(self) -> None:
    """Copy the (unmerged) base weight to CPU once. Call at startup or any time
    before the first merge; it is a no-op afterwards."""
    if self._pristine_weight is None:
      assert not self.is_merged, "cannot snapshot after a merge"
      self._pristine_weight = self.base_layer.weight.detach().to("cpu", copy=True)

  def merge_adapter(self, A: torch.Tensor, B: torch.Tensor, scale: float) -> None:
    """Fold one adapter into the base weight: W += scale * B @ A (fp32 math,
    single bf16 rounding). Requires a pristine snapshot for exact restore."""
    self.snapshot_pristine()
    w = self.base_layer.weight.data
    delta = (B.detach().float() * scale) @ A.detach().float()
    w.copy_((w.float() + delta).to(w.dtype))
    self.is_merged = True
    self.weights_version += 1

  def restore_pristine(self) -> None:
    """Bitwise-restore the base weight from the CPU snapshot."""
    if not self.is_merged:
      return
    assert self._pristine_weight is not None
    self.base_layer.weight.data.copy_(self._pristine_weight, non_blocking=True)
    self.is_merged = False
    self.weights_version += 1

  def clear_runtime_lora(self) -> None:
    """Drop the runtime adapter stack (merged weights are unaffected)."""
    self.lora_weights_list.clear()
    self._lora_cache = None
    self.disable_lora = True

  @property
  def disable_lora(self) -> bool:
    return self._disable_lora

  @disable_lora.setter
  def disable_lora(self, value: bool) -> None:
    if value != self._disable_lora:
      self.weights_version += 1
    self._disable_lora = value

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
    self.weights_version += 1

  def deactivate(self) -> None:
    self.disable_lora = True
    self._lora_cache = None

  def _build_cache(self, dtype: torch.dtype) -> None:
    # Concatenate all adapters into a single (A, B) pair along the rank dim so the
    # forward pays two skinny GEMMs total instead of two per adapter. Exact:
    # B_cat @ (A_cat @ x) == sum_i B_i @ (A_i @ x), with the sum accumulated in
    # fp32 inside one GEMM instead of rounded to bf16 per adapter.
    a_list, b_list = [], []
    for lora_A, lora_B, _, lora_strength, lora_rank, lora_alpha in self.lora_weights_list:
      scale = lora_strength
      if lora_alpha is not None and lora_rank is not None and lora_alpha != lora_rank:
        scale *= lora_alpha / lora_rank
      a_list.append(lora_A.to(dtype=dtype))
      b_list.append((lora_B * scale).to(dtype=dtype))
    self._lora_cache = (torch.cat(a_list, dim=0), torch.cat(b_list, dim=1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    result = self.base_layer(x)

    if self.disable_lora or not self.lora_weights_list:
      return result

    if self._lora_cache is None:
      self._build_cache(x.dtype)

    lora_A_cat, scaled_lora_B_cat = self._lora_cache
    h = F.linear(x, lora_A_cat)
    out_features = result.shape[-1]
    result.view(-1, out_features).addmm_(h.view(-1, h.shape[-1]), scaled_lora_B_cat.t())
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
