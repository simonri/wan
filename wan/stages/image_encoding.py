import PIL.Image
import torch

from wan.modules.wanvae import Wan2_1_VAE
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.torch_utils import PRECISION_TO_TYPE
from wan.vision_utils import normalize, numpy_to_pt, pil_to_numpy


class ImageVAEEncodingStage(PipelineStage):
  def __init__(self, vae: Wan2_1_VAE, **kwargs):
    super().__init__()
    self.vae = vae

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

    return batch
