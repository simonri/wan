import PIL.Image

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

  def forward(self, batch: Req) -> Req:
    # ensure prompt is properly formatted
    if batch.prompt is None and batch.prompt_embeds is None:
      raise ValueError("Either prompt or prompt_embeds must be provided")

    # val infer steps
    if batch.num_inference_steps <= 0:
      raise ValueError(f"Number of inferense steps must be positive, but got {batch.num_inference_steps}")

    if batch.image_path is not None:
      image = load_image(batch.image_path)
      batch.condition_image = image
      batch.original_condition_image_size = image.size

    return batch
