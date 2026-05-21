from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class LatentPreparationStage(PipelineStage):
  def __init__(self, scheduler: FlowUniPCMultistepScheduler):
    super().__init__()
    self.scheduler = scheduler

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    return batch
