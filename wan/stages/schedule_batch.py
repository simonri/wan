from dataclasses import dataclass

import PIL.Image
import torch


@dataclass(init=False)
class Req:
  condition_image: torch.Tensor | PIL.Image.Image | None = None
  vae_image: torch.Tensor | PIL.Image.Image | None = None
