from dataclasses import dataclass
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from wan.platform import get_local_torch_device
from wan.schedulers.base import BaseScheduler
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


@dataclass(frozen=True)
class LatentPreparationFingerprint:
  height: int | None
  width: int | None
  num_frames: int | None
  latent_num_frames: int | None
  prompt_dtype: Any
  generator_device: str | None


class LatentPreparationStage(PipelineStage):
  def __init__(self, scheduler: BaseScheduler):
    super().__init__()
    self.scheduler = scheduler

  def _get_latent_dtype(
    self,
    batch: Req,
    server_args: ServerArgs,
  ):
    return server_args.pipeline_config.get_latent_dtype(batch.prompt_embeds[0].dtype)

  @staticmethod
  def _single_generator(batch: Req):
    if isinstance(batch.generator, list):
      assert len(batch.generator) == 1
      return batch.generator[0]
    return batch.generator

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

  def build_dedup_fingerprint(
    self,
    batch: Req,
    server_args: ServerArgs,
  ) -> LatentPreparationFingerprint:
    prompt_dtype = batch.prompt_embeds[0].dtype if isinstance(batch.prompt_embeds, list) else batch.prompt_embeds.dtype
    latent_num_frames = self.adjust_video_length(batch, server_args)
    return LatentPreparationFingerprint(
      height=batch.height,
      width=batch.width,
      num_frames=batch.num_frames,
      latent_num_frames=latent_num_frames,
      prompt_dtype=prompt_dtype,
      generator_device=batch.generator_device,
    )

  def _prepare_grouped_latents(
    self,
    batches: list[Req],
    server_args: ServerArgs,
  ) -> Req:
    """Prepare grouped random latents without changing per-request RNG streams.

    ``randn_tensor`` accepts a list of generators, but its batched draw is not
    guaranteed to match drawing each request independently. For multi-output
    requests we need exact equivalence to the sequential seed path, so this
    helper draws one raw latent tensor per request and only batches the
    deterministic packing/scaling work.
    """
    first_batch = batches[0]
    latent_num_frames = self.adjust_video_length(first_batch, server_args)
    batch_size = len(batches)

    dtype = self._get_latent_dtype(first_batch, server_args)
    device = get_local_torch_device()
    num_frames = latent_num_frames if latent_num_frames is not None else first_batch.num_frames
    height = first_batch.height
    width = first_batch.width

    if height is None or width is None:
      raise ValueError("Height and width must be provided")

    raw_latents = []
    for batch in batches:
      shape = server_args.pipeline_config.prepare_latent_shape(batch, 1, num_frames)
      raw_latents.append(
        randn_tensor(
          shape,
          generator=self._single_generator(batch),
          device=device,
          dtype=dtype,
        )
      )

    latents = torch.cat(raw_latents, dim=0)

    original_num_outputs = first_batch.num_outputs_per_prompt
    try:
      first_batch.num_outputs_per_prompt = batch_size
    finally:
      first_batch.num_outputs_per_prompt = original_num_outputs

    if hasattr(self.scheduler, "init_noise_sigma"):
      latents = latents * self.scheduler.init_noise_sigma

    if hasattr(self.scheduler, "init_noise_sigma"):
      latents = latents * self.scheduler.init_noise_sigma

    first_batch.latents = latents
    first_batch.raw_latent_shape = latents.shape
    return first_batch

  def _split_batched_latents(self, src: Req, batches: list[Req]) -> None:
    total = len(batches)
    assert src.latents is not None
    latents = src.latents
    for index, batch in enumerate(batches):
      batch.latents = self._slice_batch_tensor(latents, index, total)
      batch.raw_latent_shape = batch.latents.shape

  @staticmethod
  def _slice_batch_tensor(tensor: torch.Tensor, index: int, total: int):
    if tensor.shape[0] == total:
      return tensor[index : index + 1].contiguous()
    return tensor

  def run_grouped_requests(
    self,
    batches: list[Req],
    server_args: ServerArgs,
  ) -> list[Req]:
    results: list[Req | None] = [None] * len(batches)

    for _, group in self._group_requests_by_fingerprint(
      batches, lambda batch: self.build_dedup_fingerprint(batch, server_args)
    ):
      indexed_batches = group
      group_batches = [batch for _, batch in indexed_batches]
      if len(group_batches) == 1 or any(batch.latents is not None for batch in group_batches):
        for index, batch in indexed_batches:
          results[index] = self(batch, server_args)
        continue

      first_result = self._prepare_grouped_latents(group_batches, server_args)
      self._split_batched_latents(first_result, group_batches)
      for index, batch in indexed_batches:
        results[index] = batch

    return results
