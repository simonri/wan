from diffusers.utils.torch_utils import randn_tensor

from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class LatentPreparationStage(PipelineStage):
  def __init__(self, scheduler: FlowUniPCMultistepScheduler):
    super().__init__()
    self.scheduler = scheduler

  def adjust_video_length(self, batch: Req, server_args: ServerArgs) -> int:
    video_length = batch.num_frames
    latent_num_frames = video_length
    use_temporal_scaling_frames = server_args.pipeline_config.vae_config.use_temporal_scaling_frames
    if use_temporal_scaling_frames:
      temporal_scale_factor = server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
      latent_num_frames = (video_length - 1) // temporal_scale_factor + 1
    return int(latent_num_frames)

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    latent_num_frames = self.adjust_video_length(batch, server_args)

    batch_size = batch.batch_size

    dtype = batch.prompt_embeds[0].dtype
    device = get_local_torch_device()
    generator = batch.generator
    latents = batch.latents
    num_frames = latent_num_frames if latents is None else batch.num_frames

    if latents is None:
      shape = server_args.pipeline_config.prepare_latent_shape(batch, batch_size, num_frames)
      latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
      latents = latents.to(device)

    batch.latents = latents
    batch.raw_latent_shape = latents.shape

    return batch
