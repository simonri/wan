from dataclasses import MISSING, dataclass, field, fields
from typing import Any

import PIL.Image
import torch

from wan.configs.sample.base import SamplingParams

SAMPLING_PARAMS_FIELDS = {f.name for f in fields(SamplingParams)}


@dataclass(init=False)
class Req:
  """
  Complete state passed through the pipeline execution.
  """

  sampling_params: SamplingParams | None = None

  # image encoder hidden states
  image_embeds: list[torch.Tensor] = field(default_factory=list)

  original_condition_image_size: tuple[int, int] = None
  condition_image: torch.Tensor | PIL.Image.Image | None = None
  vae_image: torch.Tensor | PIL.Image.Image | None = None

  # primary encoder embeddings
  prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
  prompt_attention_mask: list[torch.Tensor | None] | None = None
  prompt_embeds_mask: list[torch.Tensor | None] | None = None
  prompt_seq_lens: list[list[int]] | None = None

  pooled_embeds: list[torch.Tensor] = field(default_factory=list)

  # additional text-related parameters
  do_classifier_free_guidance: bool = False

  def __init__(self, **kwargs):
    for name, value in self.__class__.__dataclass_fields__.items():
      if name in kwargs:
        object.__setattr__(self, name, kwargs.pop(name))
      elif value.default is not MISSING:
        object.__setattr__(self, name, value.default)
      elif value.default_factory is not MISSING:
        object.__setattr__(self, name, value.default_factory())

    for name, value in kwargs.items():
      setattr(self, name, value)

  def __getattr__(self, name: str) -> any:
    """
    Delegate attribute access to sampling_params if not found in Req.
    This is only called when the attribute is not found in the instance.
    """
    if name == "sampling_params":
      raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    sampling_params = object.__getattribute__(self, "sampling_params")
    if sampling_params is not None and hasattr(sampling_params, name):
      return getattr(sampling_params, name)

    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

  def __set_attr__(self, name: str, value: any) -> None:
    if name == "sampling_params":
      object.__setattr__(self, name, value)

    if name in self.__class__.__dataclass_fields__:
      object.__setattr__(self, name, value)
      return

    try:
      sampling_params = object.__getattribute__(self, "sampling_params")
    except AttributeError:
      sampling_params = None

    if sampling_params is not None and hasattr(sampling_params, name):
      setattr(sampling_params, name, value)
      return

    if sampling_params is None and name in SAMPLING_PARAMS_FIELDS:
      new_sp = SamplingParams()
      object.__setattr__(self, "sampling_params", new_sp)
      setattr(new_sp, name, value)
      return

    object.__setattr__(self, name, value)


@dataclass
class OutputBatch:
  output: Any | None = None
