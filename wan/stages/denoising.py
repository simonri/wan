from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from wan.platform import get_local_torch_device
from wan.profiler import DiffusionProfiler
from wan.schedulers.base import BaseScheduler
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.torch_utils import PRECISION_TO_TYPE


@dataclass(slots=True)
class DenoisingContext:
  scheduler: Any
  extra_step_kwargs: dict[str, Any]
  target_dtype: torch.dtype
  autocast_enabled: bool
  timesteps: torch.Tensor
  num_inference_steps: int
  num_warmup_steps: int
  latents: torch.Tensor
  prompt_embeds: torch.Tensor
  boundary_timestep: float | None
  z: torch.Tensor | None
  reserved_frames_masks: torch.Tensor | None
  seq_len: int | None
  guidance: torch.Tensor
  is_warmup: bool

  def __getitem__(self, key: str) -> Any:
    return getattr(self, key)

  def get(self, key: str, default: Any = None) -> Any:
    return getattr(self, key, default)


@dataclass(slots=True)
class DenoisingStepState:
  step_index: int
  t_host: torch.Tensor
  t_device: torch.Tensor
  t_int: int
  current_model: Any
  current_guidance_scale: Any


class DenoisingStage(PipelineStage):
  def __init__(
    self,
    transformer,
    scheduler: BaseScheduler,
    transformer_2=None,
  ):
    super().__init__()
    self.transformer = transformer
    self.transformer_2 = transformer_2
    self.scheduler = scheduler

  def _handle_boundary_ratio(self, server_args: ServerArgs, batch: Req, scheduler: BaseScheduler) -> float:
    boundary_ratio = server_args.pipeline_config.dit_config.boundary_ratio
    if batch.boundary_ratio is not None:
      print(f"Overriding boundary ratio from {boundary_ratio} to {batch.boundary_ratio}")
      boundary_ratio = batch.boundary_ratio

    if boundary_ratio is not None:
      num_train_timesteps = scheduler.config.num_train_timesteps
      boundary_timestep = boundary_ratio * num_train_timesteps
    else:
      boundary_timestep = None

    return boundary_timestep

  def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs) -> DenoisingContext:
    assert self.transformer is not None
    scheduler = batch.scheduler
    assert scheduler is not None

    boundary_timestep = self._handle_boundary_ratio(server_args, batch, scheduler)
    timesteps = batch.timesteps
    num_inference_steps = batch.num_inference_steps
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

    target_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
    autocast_enabled = target_dtype != torch.float32

    # prepare image latents and embeddings
    image_embeds = batch.image_embeds
    if len(image_embeds) > 0:
      image_embeds = [image_embed.to(target_dtype) for image_embed in image_embeds]

    seq_len, z, reserved_frames_masks = (None, None, None)

    reserved_frames_masks, z_sp = (reserved_frames_masks[0] if reserved_frames_masks is not None else None,), z

    latents = batch.latents

    # note - guidance will be None if cfg is 1.0
    guidance = None

    # convert once so every step passes the same tensor object to the model,
    # letting the model's per-context caches (text embed, cross-attn K/V) hit
    prompt_embeds = batch.prompt_embeds.to(target_dtype)

    return DenoisingContext(
      scheduler=scheduler,
      extra_step_kwargs={},
      timesteps=timesteps,
      latents=latents,
      prompt_embeds=prompt_embeds,
      boundary_timestep=boundary_timestep,
      guidance=guidance,
      z=z_sp,
      seq_len=seq_len,
      reserved_frames_masks=reserved_frames_masks,
      autocast_enabled=autocast_enabled,
      target_dtype=target_dtype,
      num_inference_steps=num_inference_steps,
      num_warmup_steps=num_warmup_steps,
      is_warmup=batch.is_warmup,
    )

  def _select_and_manage_model(
    self,
    t_int: int,
    boundary_timestep: float | None,
    server_args: ServerArgs,
    batch: Req,
  ):
    if boundary_timestep is None or t_int >= boundary_timestep:
      # high noise stage
      current_model = self.transformer
      current_guidance_scale = batch.guidance_scale
    else:
      # low noise stage
      current_model = self.transformer_2
      current_guidance_scale = batch.guidance_scale_2

    assert current_model is not None, "The model for the current step is not set"
    assert current_model, current_guidance_scale

    return current_model, current_guidance_scale

  def _prepare_step_state(
    self,
    ctx: DenoisingContext,
    batch: Req,
    server_args: ServerArgs,
    step_index: int,
    t_host: torch.Tensor,
    timesteps_cpu: torch.Tensor,
  ) -> DenoisingStepState:
    t_int = int(t_host.item())
    t_device = ctx.timesteps[step_index]

    current_model, current_guidance_scale = self._select_and_manage_model(
      t_int=t_int,
      boundary_timestep=ctx.boundary_timestep,
      server_args=server_args,
      batch=batch,
    )

    return DenoisingStepState(
      step_index=step_index,
      t_host=t_host,
      t_device=t_device,
      t_int=t_int,
      current_model=current_model,
      current_guidance_scale=current_guidance_scale,
    )

  def expand_timestep_before_forward(
    self, batch: Req, server_args: ServerArgs, t_device, target_dtype, seq_len: int | None, reserved_frames_masks
  ):
    bsz = batch.raw_latent_shape[0]
    timestep = t_device.repeat(bsz)
    return timestep

  def _predict_noise(
    self,
    current_model: nn.Module,
    latent_model_input: torch.Tensor,
    timestep,
    encoder_hidden_states,
  ) -> torch.Tensor:
    return current_model(
      hidden_states=latent_model_input,
      timestep=timestep,
      encoder_hidden_states=encoder_hidden_states,
    )

  def _run_denoising_step(
    self,
    ctx: DenoisingContext,
    step: DenoisingStepState,
    batch: Req,
    server_args: ServerArgs,
  ) -> None:
    # 1. prepare latent inputs in the models compute dtype
    latent_model_input = ctx.latents.to(ctx.target_dtype)

    if batch.image_latent is not None:
      latent_model_input = torch.cat([latent_model_input, batch.image_latent], dim=1).to(ctx.target_dtype)

    # 2. expand the timestep to the shape expected by the current model
    timestep = self.expand_timestep_before_forward(
      batch,
      server_args,
      step.t_device,
      ctx.target_dtype,
      ctx.seq_len,
      ctx.reserved_frames_masks,
    )

    # 3. apply scheduler side input scaling before the model forward
    latent_model_input = ctx.scheduler.scale_model_input(latent_model_input, step.t_device)

    # 4. run the model prediction path
    noise_pred = self._predict_noise(
      current_model=step.current_model,
      latent_model_input=latent_model_input,
      timestep=timestep,
      encoder_hidden_states=ctx.prompt_embeds,
    )

    # 5. advance the scheduler state with the predicted noise
    ctx.latents = ctx.scheduler.step(
      model_output=noise_pred,
      timestep=step.t_device,
      sample=ctx.latents,
      return_dict=False,
    )[0]

  def _before_denoising_loop(self, ctx: DenoisingContext, batch: Req, server_args: ServerArgs):
    self._reset_scheduler_loop_state(ctx.scheduler)
    ctx.scheduler.set_begin_index(0)

  def _reset_scheduler_loop_state(self, scheduler) -> None:
    if hasattr(scheduler, "_step_index"):
      scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
      scheduler._begin_index = None
    if hasattr(scheduler, "lower_order_nums"):
      scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
      scheduler.last_sample = None
    if hasattr(scheduler, "this_order"):
      scheduler.this_order = 0

    solver_order = getattr(getattr(scheduler, "config", None), "solver_order", 0)
    if solver_order:
      if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * solver_order
      if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * solver_order

  def _post_denoising_loop(self, batch: Req, latents: torch.Tensor):
    batch.latents = latents

  def progress_bar(self, iterable: Iterable | None = None, total: int | None = None):
    return tqdm(iterable, total=total)

  def step_profile(self):
    profiler = DiffusionProfiler.get_instance()
    if profiler:
      profiler.step_denoising_step()

  @torch.no_grad()
  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    ctx = self._prepare_denoising_loop(batch, server_args)

    self._before_denoising_loop(ctx, batch, server_args)

    local_device = get_local_torch_device()

    timesteps_cpu = ctx.timesteps.cpu()

    with torch.autocast(
      device_type=local_device.type,
      dtype=ctx.target_dtype,
      enabled=ctx.autocast_enabled,
    ):
      num_timesteps = timesteps_cpu.shape[0]

      with self.progress_bar(total=ctx.num_inference_steps) as progress_bar:
        for step_index, t_host in enumerate(timesteps_cpu):
          step = self._prepare_step_state(
            ctx,
            batch,
            server_args,
            step_index,
            t_host,
            timesteps_cpu,
          )

          self._run_denoising_step(ctx, step, batch, server_args)

          if step_index == num_timesteps - 1 or (
            (step_index + 1) > ctx.num_warmup_steps and (step_index + 1) % ctx.scheduler.order == 0
          ):
            progress_bar.update()

          if not ctx.is_warmup:
            self.step_profile()

    self._post_denoising_loop(batch, ctx.latents)
    return batch
