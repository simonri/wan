import torch
import torch.nn as nn

from wan.platform import get_local_torch_device

LORA_MERGE_CHUNK_BYTES = 32 * 1024 * 1024
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
    # Pristine weight backup for exact restore on unmerge. Lazily snapshotted on first
    # merge so layers that never receive a LoRA never pay the GPU->CPU copy.
    self.cpu_weight: torch.Tensor | None = None

    self.disable_lora: bool = True
    self.lora_rank = lora_rank
    self.lora_alpha = lora_alpha
    self.lora_weights_list: list[LoRAWeightEntry] = []

  def _ensure_cpu_weight_snapshot(self) -> None:
    if self.cpu_weight is None:
      self.cpu_weight = self.base_layer.weight.detach().to("cpu").clone()

  @property
  def weight(self) -> torch.Tensor:
    return self.base_layer.weight

  @property
  def bias(self) -> torch.Tensor:
    return self.base_layer.bias

  def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
    return A

  def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
    return B

  @staticmethod
  def _as_mutable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_inference():
      with torch.inference_mode(False):
        return tensor.detach().clone()
    return tensor

  @torch.no_grad()
  def _merge_lora_into_data(
    self,
    data: torch.Tensor,
    lora_list: list[LoRAWeightEntry],
  ) -> None:
    # Merge all LoRA adapters in order
    for lora_A, lora_B, _, lora_strength, lora_rank, lora_alpha in lora_list:
      lora_A_sliced = self.slice_lora_a_weights(lora_A.to(data))
      lora_B_sliced = self.slice_lora_b_weights(lora_B.to(data))

      scale = lora_strength
      if lora_alpha is not None and lora_rank is not None and lora_alpha != lora_rank:
        scale *= lora_alpha / lora_rank

      if not isinstance(lora_B_sliced, torch.Tensor):
        lora_delta = lora_B_sliced @ lora_A_sliced
        if isinstance(lora_delta, torch.Tensor) and lora_delta.dim() > 2:
          lora_delta = lora_delta.reshape(-1, lora_delta.shape[-1])
        data.add_(lora_delta, alpha=scale)
        continue

      if lora_A_sliced.dim() > 2 or lora_B_sliced.dim() > 2:
        lora_delta = lora_B_sliced @ lora_A_sliced
        if lora_delta.dim() > 2:
          lora_delta = lora_delta.reshape(-1, lora_delta.shape[-1])
        data_2d = data.reshape(-1, data.shape[-1]) if data.dim() > 2 else data
        data_2d.add_(lora_delta, alpha=scale)
        continue

      data_2d = data.reshape(-1, data.shape[-1]) if data.dim() > 2 else data
      lora_B_2d = lora_B_sliced.reshape(-1, lora_B_sliced.shape[-1]) if lora_B_sliced.dim() > 2 else lora_B_sliced

      chunk_rows = max(
        1,
        LORA_MERGE_CHUNK_BYTES // (data_2d.shape[-1] * max(1, data_2d.element_size())),
      )
      for start in range(0, lora_B_2d.shape[0], chunk_rows):
        end = min(start + chunk_rows, lora_B_2d.shape[0])
        chunk_delta = lora_B_2d[start:end] @ lora_A_sliced
        data_2d[start:end].add_(chunk_delta, alpha=scale)

  @torch.no_grad()
  def merge_lora_weights(self) -> None:
    if self.disable_lora:
      return

    if self.merged:
      self.unmerge_lora_weights()

    lora_list = self.lora_weights_list
    if not lora_list:
      raise ValueError("LoRA weights not set. Please set them first.")

    # snapshot the pristine weight on first merge so unmerge can restore it exactly
    self._ensure_cpu_weight_snapshot()

    current_device = self.base_layer.weight.data.device
    data = self.base_layer.weight.data.to(get_local_torch_device())
    data = self._as_mutable_tensor(data)
    target_dtype = data.dtype

    self._merge_lora_into_data(data, lora_list)

    self.base_layer.weight.data = self._as_mutable_tensor(
      data.to(current_device, dtype=target_dtype, non_blocking=True)
    )

    self.merged = True

  @torch.no_grad()
  def unmerge_lora_weights(self) -> None:
    if self.disable_lora:
      return

    if not self.merged:
      raise ValueError("LoRA weights are not merged. Please merge them first.")

    current_device = self.base_layer.weight.data.device
    cpu_weight_on_device = self.cpu_weight.to(current_device, non_blocking=True)
    if self.base_layer.weight.data.is_inference():
      self.base_layer.weight.data = self._as_mutable_tensor(cpu_weight_on_device)
    else:
      self.base_layer.weight.data.copy_(cpu_weight_on_device)
    if cpu_weight_on_device.data_ptr() != self.base_layer.weight.data.data_ptr():
      del cpu_weight_on_device

    self.merged = False

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


class LinearWithLoRA(BaseLayerWithLoRA):
  def __init__(
    self,
    base_layer: nn.Linear,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
  ) -> None:
    super().__init__(base_layer, lora_rank, lora_alpha)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # LoRA is always merged into base_layer.weight (or unmerged back to pristine
    # on deactivate), so the forward is just the base linear.
    return self.base_layer(x)


def wrap_with_lora_layer(
  layer: nn.Module,
  lora_rank: int | None = None,
  lora_alpha: int | None = None,
) -> nn.Module | None:
  """
  Transform the given layer to its corresponding LoRA layer.
  """
  supported_layer_types: dict[type[nn.Linear]] = {nn.Linear: LinearWithLoRA}

  for src_layer_type, lora_layer_type in supported_layer_types.items():
    if isinstance(layer, src_layer_type):
      ret = lora_layer_type(
        layer,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
      )
      return ret
  return None


# source: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/lora/utils.py#L9
def replace_submodule(model: nn.Module, module_name: str, new_module: nn.Module) -> nn.Module:
  """Replace a submodule in a model with a new module."""
  parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
  target_name = module_name.split(".")[-1]
  setattr(parent, target_name, new_module)
  return new_module
