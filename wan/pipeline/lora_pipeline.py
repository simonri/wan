from collections import defaultdict

import torch
from safetensors.torch import load_file as safetensors_load_file

from wan.pipeline.base import PipelineBase
from wan.platform import get_local_torch_device


class LoRAPipeline(PipelineBase):
  lora_adapters: dict[str, dict[str, torch.Tensor]]  # nickname, target_weight_name -> weight
  loaded_adapter_paths: dict[str, str]  # nickname -> lora_path

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self.lora_adapters = defaultdict(dict)
    self.loaded_adapter_paths = {}

    self.device = get_local_torch_device()

  def load_lora_adapter(self, lora_path: str, lora_nickname: str):
    raw_state_dict = safetensors_load_file(lora_path)

    for name, weight in raw_state_dict.items():
      self.lora_adapters[lora_nickname][name] = weight.to(self.device)

  def set_lora(
    self,
    lora_nickname: str | None | list[str | None] = None,
    lora_path: str | None | list[str | None] = None,
  ):
    """
    Load LoRA into the pipeline and apply them to the specified transformer
    """

    for nickname, path in zip(lora_nickname, lora_path, strict=True):
      if nickname not in self.lora_adapters and path is None:
        raise ValueError(f"Adapter {nickname} not found in pipeline. Please provide lora_path to load it!")

      should_load = False
      if path is not None:
        if nickname not in self.loaded_adapter_paths:
          should_load = True
        elif self.loaded_adapter_paths[nickname] != path:
          should_load = True

      if should_load:
        self.load_lora_adapter(path, nickname)
