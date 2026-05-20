import numpy as np
import PIL.Image
import torch


def pil_to_numpy(images: list[PIL.Image.Image] | PIL.Image.Image) -> np.ndarray:
  if isinstance(images, PIL.Image.Image):
    images = [images]

  images = [np.array(image).astype(np.float32) / 255.0 for image in images]
  images_arr: np.ndarray = np.stack(images, axis=0)
  return images_arr


def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
  if images.ndim == 3:
    images = images[..., None]

  images = torch.from_numpy(images.transpose(0, 3, 1, 2))
  return images


def normalize(images: torch.Tensor) -> torch.Tensor:
  """normalize image array to [-1, 1]"""
  return 2.0 * images - 1.0
