from copy import deepcopy
from typing import Any

from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def get_or_create_request_scheduler(
  batch: Req,
  scheduler_template: Any,
  *,
  isolate: bool = False,
) -> Any:
  if batch.scheduler is None:
    batch.scheduler = deepcopy(scheduler_template) if isolate else scheduler_template

  return batch.scheduler


class TimestepPreparationStage(PipelineStage):
  def __init__(self, scheduler: FlowUniPCMultistepScheduler):
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
