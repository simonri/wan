import os

import numpy as np
import PIL.Image
import torch

from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req

# Decoded-image cache: streams re-submit the same reference image for every clip,
# and decoding a multi-megapixel photo costs ~150ms per job. Keyed by
# (path, mtime) so an edited file is picked up. Callers must not mutate the
# returned image in place (resize/convert return new images, so the existing
# pipeline is safe).
_IMAGE_CACHE: dict[tuple[str, float | None], PIL.Image.Image] = {}
_IMAGE_CACHE_MAX = 16


def load_image(
  image: str,
) -> PIL.Image.Image:
  try:
    mtime = os.path.getmtime(image)
  except OSError:
    mtime = None
  key = (image, mtime)
  cached = _IMAGE_CACHE.get(key)
  if cached is not None:
    return cached

  loaded = PIL.Image.open(image)
  loaded = PIL.ImageOps.exif_transpose(loaded)
  loaded = loaded.convert("RGB")
  loaded.load()

  if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
    _IMAGE_CACHE.clear()
  _IMAGE_CACHE[key] = loaded
  return loaded


def resize_and_center_crop(image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
  """Scale `image` to cover (width, height) preserving aspect ratio, then
  center-crop the overflow. Unlike a bare .resize(), this never stretches the
  source photo when its aspect ratio differs from the target canvas."""
  src_w, src_h = image.size
  scale = max(width / src_w, height / src_h)
  scaled_w, scaled_h = round(src_w * scale), round(src_h * scale)
  image = image.resize((scaled_w, scaled_h), PIL.Image.LANCZOS)

  left = (scaled_w - width) // 2
  top = (scaled_h - height) // 2
  return image.crop((left, top, left + width, top + height))


class InputValidationStage(PipelineStage):
  def __init__(self):
    super().__init__()

  def _generate_seeds(self, batch: Req, server_args: ServerArgs):
    seed = batch.seed
    num_videos_per_prompt = batch.num_outputs_per_prompt

    assert seed is not None

    prompt_count = len(batch.prompt) if isinstance(batch.prompt, list) else 1

    # todo: support list of seeds?
    base_seeds = [int(seed) + 1 * num_videos_per_prompt * i for i in range(prompt_count)]
    seeds = []
    for base_seed in base_seeds:
      seeds.extend([base_seed + i for i in range(num_videos_per_prompt)])

    batch.seeds = seeds

    generator_device = batch.generator_device
    if generator_device is None:
      generator_device = getattr(server_args.pipeline_config, "generator_device", None) or get_local_torch_device().type

    batch.generator = [torch.Generator(device=generator_device).manual_seed(seed) for seed in seeds]

  @staticmethod
  def _calculate_dimensions_from_area(max_area: float, aspect_ratio: float, mod_value: int) -> tuple[int, int]:
    height = round(np.sqrt(max_area * aspect_ratio) // mod_value * mod_value)
    width = round(np.sqrt(max_area / aspect_ratio) // mod_value * mod_value)
    return width, height

  def preprocess_condition_image(
    self,
    batch: Req,
    server_args: ServerArgs,
    condition_image_width,
    condition_image_height,
  ):
    max_area = server_args.pipeline_config.max_area
    image_aspect_ratio = condition_image_height / condition_image_width
    mod_value = (
      server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
      * server_args.pipeline_config.dit_config.arch_config.patch_size[1]
    )

    if batch.width is not None and batch.height is not None:
      # Caller pinned an exact canvas (e.g. every streaming session): honor
      # ITS aspect ratio, not the photo's. Otherwise the mod_value-rounded
      # output silently drifts off the requested size whenever the photo's
      # aspect ratio isn't exactly height/width (virtually always, for a real
      # photo) — producing frames a few rows off from the idle clip / static
      # canvas the rest of the session assumes, visible as a torn bottom edge.
      aspect_ratio = batch.height / batch.width
      target_area = min(batch.width * batch.height, max_area)
      if batch.width * batch.height > max_area:
        print(f"Warning: image area {batch.width * batch.height} is greater than max area {max_area}")
    elif batch.width is not None or batch.height is not None:
      aspect_ratio = image_aspect_ratio
      if batch.width is None:
        batch.width = round(batch.height / aspect_ratio)
      elif batch.height is None:
        batch.height = round(batch.width * aspect_ratio)

      target_area = min(batch.width * batch.height, max_area)
      if batch.width * batch.height > max_area:
        print(f"Warning: image area {batch.width * batch.height} is greater than max area {max_area}")

    else:
      aspect_ratio = image_aspect_ratio
      target_area = max_area
    width, height = self._calculate_dimensions_from_area(target_area, aspect_ratio, mod_value)

    batch.condition_image = resize_and_center_crop(batch.condition_image, width, height)
    batch.height = height
    batch.width = width

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    self._generate_seeds(batch, server_args)

    # ensure prompt is properly formatted
    if batch.prompt is None and batch.prompt_embeds is None:
      raise ValueError("Either prompt or prompt_embeds must be provided")

    # val infer steps
    if batch.num_inference_steps <= 0:
      raise ValueError(f"Number of inferense steps must be positive, but got {batch.num_inference_steps}")

    if batch.image_path is not None:
      image = load_image(batch.image_path)
      batch.condition_image = image
      condition_image_width, condition_image_height = (image.width, image.height)
      batch.original_condition_image_size = image.size

      self.preprocess_condition_image(batch, server_args, condition_image_width, condition_image_height)

    end_image_path = getattr(batch, "end_image_path", None)
    if end_image_path is not None:
      end_image = load_image(end_image_path)
      batch.end_condition_image = resize_and_center_crop(end_image, batch.width, batch.height)

    return batch
