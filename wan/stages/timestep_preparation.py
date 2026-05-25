from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from wan.platform import get_local_torch_device
from wan.schedulers.base import BaseScheduler
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


def get_or_create_request_scheduler(
  batch: Req,
  scheduler_template: Any,
  *,
  isolate: bool = False,
) -> Any:
  if batch.scheduler is None:
    batch.scheduler = deepcopy(scheduler_template) if isolate else scheduler_template

  return batch.scheduler


@dataclass(frozen=True)
class TimestepPreparationFingerprint:
  num_inference_steps: int
  timesteps: Any
  sigmas: Any
  n_tokens: int | None
  height: int | None
  width: int | None
  num_frames: int | None


class TimestepPreparationStage(PipelineStage):
  deduplicated_tensor_tree_output_fields = ("timesteps",)
  deduplicated_deepcopy_output_fields = ("scheduler",)

  def __init__(self, scheduler: BaseScheduler):
    super().__init__()
    self.scheduler = scheduler

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    if batch.scheduler is not None and batch.timesteps is not None:
      return batch

    scheduler = get_or_create_request_scheduler(batch, self.scheduler)
    num_inference_steps = batch.num_inference_steps
    device = get_local_torch_device()

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    batch.timesteps = timesteps
    batch.scheduler = scheduler

    return batch

  def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs) -> TimestepPreparationFingerprint:
    return TimestepPreparationFingerprint(
      num_inference_steps=batch.num_inference_steps,
      timesteps=self.freeze_for_dedup(batch.timesteps),
      sigmas=self.freeze_for_dedup(batch.sigmas),
      n_tokens=batch.n_tokens,
      height=batch.height,
      width=batch.width,
      num_frames=batch.num_frames,
    )
