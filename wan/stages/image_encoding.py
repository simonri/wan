import PIL.Image
import torch

from wan.modules.vae2_1 import Wan2_1_VAE
from wan.platform import get_local_torch_device
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
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

  def forward(self, batch: Req) -> Req:
    if batch.condition_image is None:
      return batch

    num_frames = batch.num_frames

    images = batch.vae_image if batch.vae_image is not None else batch.condition_image
    if not isinstance(images, list):
      images = [images]

    for image in images:
      local_device = get_local_torch_device()

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

      video_condition = video_condition.to(local_device, dtype=torch.float32)

      # encode image
      latent_dist = self.vae.encode(video_condition)
      print(latent_dist.shape)
