import numpy as np
import PIL.Image

from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


def load_image(
  image: str,
) -> PIL.Image.Image:
  image = PIL.Image.open(image)
  image = PIL.ImageOps.exif_transpose(image)
  return image


class InputValidationStage(PipelineStage):
  def __init__(self):
    super().__init__()

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
    aspect_ratio = condition_image_height / condition_image_width
    mod_value = (
      server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
      * server_args.pipeline_config.dit_config.arch_config.patch_size[1]
    )

    if batch.width is not None or batch.height is not None:
      if batch.width is None:
        batch.width = round(batch.height / aspect_ratio)
      elif batch.height is None:
        batch.height = round(batch.width * aspect_ratio)

      target_area = min(batch.width * batch.height, max_area)
      if batch.width * batch.height > max_area:
        print(f"Warning: image area {batch.width * batch.height} is greater than max area {max_area}")

    else:
      target_area = max_area
    width, height = self._calculate_dimensions_from_area(target_area, aspect_ratio, mod_value)

    batch.condition_image = batch.condition_image.resize((width, height))
    batch.height = height
    batch.width = width

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
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

    return batch
