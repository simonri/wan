from dataclasses import dataclass
from typing import Any

import PIL.Image
import torch

from wan.modules.wanvae import Wan2_1_VAE
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.torch_utils import PRECISION_TO_TYPE
from wan.vision_utils import normalize, numpy_to_pt, pil_to_numpy


@dataclass(frozen=True)
class ImageVAEEncodingFingerprint:
  image_source: Any
  end_image_source: Any
  height: int | None
  width: int | None
  num_frames: int | None
  vae_precision: Any


def _freeze_image_source_value(value):
  """Build a hashable identity fragment for image inputs.

  Image inputs are often PIL/numpy/tensor objects. For file paths we can use
  the path value; for in-memory objects we only dedup when the exact same
  object instance is shared by multiple requests. This avoids expensive image
  hashing and avoids treating two mutable image objects as equivalent just
  because they currently have the same shape.
  """
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_image_source_value(item) for item in value)
  if isinstance(value, (str, int, float, bool, type(None))):
    return value
  return ("object", id(value))


def _build_image_source_fingerprint(batch: Req, *, prefer_vae_image: bool = False):
  """Return the image input fragment used by image encoding fingerprints."""
  if batch.image_path is not None:
    return ("path", PipelineStage.freeze_for_dedup(batch.image_path))
  image = batch.vae_image if prefer_vae_image and batch.vae_image is not None else None
  if image is None:
    image = batch.condition_image
  return ("image", _freeze_image_source_value(image))


class ImageVAEEncodingStage(PipelineStage):
  deduplicated_output_fields = ("image_latent",)

  def __init__(self, vae: Wan2_1_VAE, **kwargs):
    super().__init__()
    self.vae = vae
    self._cache: dict[ImageVAEEncodingFingerprint, torch.Tensor] = {}

  def preprocess(
    self,
    image: torch.Tensor | PIL.Image.Image,
  ) -> torch.Tensor:
    if isinstance(image, PIL.Image.Image):
      image = pil_to_numpy(image)
      image = numpy_to_pt(image)

    do_normalize = True
    if image.min() < 0:
      do_normalize = False
    if do_normalize:
      image = normalize(image)

    return image

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    if batch.condition_image is None:
      return batch

    if batch.image_latent is not None:
      return batch

    fingerprint = self.build_dedup_fingerprint(batch, server_args)
    if fingerprint in self._cache:
      batch.image_latent = self._cache[fingerprint].clone()
      return batch

    num_frames = batch.num_frames
    local_device = get_local_torch_device()

    images = batch.vae_image if batch.vae_image is not None else batch.condition_image
    if not isinstance(images, list):
      images = [images]

    all_image_latents = []

    vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]

    for image in images:
      image = self.preprocess(image).to(local_device, dtype=torch.float32)

      # (B, C, H, W) -> (B, C, 1, H, W)
      image = image.unsqueeze(2)

      if num_frames == 1:
        video_condition = image
      elif batch.end_condition_image is not None:
        end_image = self.preprocess(batch.end_condition_image).to(local_device, dtype=torch.float32)
        end_image = end_image.unsqueeze(2)
        video_condition = torch.cat(
          [
            image,
            image.new_zeros(
              image.shape[0],
              image.shape[1],
              num_frames - 2,
              image.shape[3],
              image.shape[4],
            ),
            end_image,
          ],
          dim=2,
        )
      else:
        video_condition = torch.cat(
          [
            image,
            image.new_zeros(
              image.shape[0],
              image.shape[1],
              num_frames - 1,
              image.shape[3],
              image.shape[4],
            ),
          ],
          dim=2,
        )

      video_condition = video_condition.to(local_device, dtype=vae_dtype)

      # encode image
      latent_dist = self.vae.encode(video_condition)

      latent_condition = latent_dist.mode()

      scaling_factor, shift_factor = server_args.pipeline_config.get_decode_scale_and_shift(
        device=latent_condition.device,
        dtype=latent_condition.dtype,
        vae=self.vae,
      )

      if isinstance(shift_factor, torch.Tensor):
        shift_factor = shift_factor.to(latent_condition.device)

      if isinstance(scaling_factor, torch.Tensor):
        scaling_factor = scaling_factor.to(latent_condition.device)

      latent_condition -= shift_factor
      latent_condition = latent_condition * scaling_factor

      image_latent = server_args.pipeline_config.postprocess_image_latent(latent_condition, batch)
      all_image_latents.append(image_latent)

    batch.image_latent = torch.cat(all_image_latents, dim=1)
    self._cache[fingerprint] = batch.image_latent.clone()

    return batch

  def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs) -> ImageVAEEncodingFingerprint | int:
    if batch.condition_image is None:
      return id(batch)

    end_image_path = getattr(batch, "end_image_path", None)
    if end_image_path is not None:
      end_image_source = ("path", PipelineStage.freeze_for_dedup(end_image_path))
    elif batch.end_condition_image is not None:
      end_image_source = ("image", _freeze_image_source_value(batch.end_condition_image))
    else:
      end_image_source = None

    return ImageVAEEncodingFingerprint(
      image_source=_build_image_source_fingerprint(batch, prefer_vae_image=True),
      end_image_source=end_image_source,
      height=batch.height,
      width=batch.width,
      num_frames=batch.num_frames,
      vae_precision=server_args.pipeline_config.vae_precision,
    )
