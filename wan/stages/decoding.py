import torch

from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import OutputBatch, Req


class DecodingStage(PipelineStage):
  def __init__(self, vae):
    super().__init__()
    self.vae = vae

  def scale_and_shift(self, latents: torch.Tensor, server_args: ServerArgs) -> torch.Tensor:
    scaling_factor, shift_factor = server_args.pipeline_config.get_decode_scale_and_shift(
      latents.device, latents.dtype, self.vae
    )

    # 1. scale
    if isinstance(scaling_factor, torch.Tensor):
      latents = latents / scaling_factor.to(latents.device, latents.dtype)
    else:
      latents = latents / scaling_factor

    # 2. apply shifting if needed
    if shift_factor is not None:
      if isinstance(shift_factor, torch.Tensor):
        latents += shift_factor.to(latents.device, latents.dtype)
      else:
        latents += shift_factor

    return latents

  @torch.no_grad()
  def decode(
    self,
    latents: torch.Tensor,
    server_args: ServerArgs,
  ) -> torch.Tensor:
    latents = latents.to(get_local_torch_device())

    # scale and shift
    latents = self.scale_and_shift(latents, server_args)

    image = self.vae.decode(latents)

    # denormalize image to [0, 1]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

  @torch.no_grad()
  def forward(
    self,
    batch: Req,
    server_args: ServerArgs,
  ) -> OutputBatch:
    frames = self.decode(batch.latents, server_args)

    output_batch = OutputBatch(output=frames)

    return output_batch
