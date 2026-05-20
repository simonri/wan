from dataclasses import MISSING, dataclass, field

import PIL.Image
import torch

from wan.configs.sample.base import SamplingParams


@dataclass(init=False)
class Req:
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
