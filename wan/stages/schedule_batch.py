from dataclasses import dataclass, field

import PIL.Image
import torch


@dataclass(init=False)
class Req:
  condition_image: torch.Tensor | PIL.Image.Image | None = None
  vae_image: torch.Tensor | PIL.Image.Image | None = None

  # primary encoder embeddings
  prompt_embeds: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
  prompt_attention_mask: list[torch.Tensor | None] | None = None
  prompt_embeds_mask: list[torch.Tensor | None] | None = None
  prompt_seq_lens: list[list[int]] | None = None

  # additional text-related parameters
  do_classifier_free_guidance: bool = False
