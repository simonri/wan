from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


class LatentPreparationStage(PipelineStage):
  def __init__(self):
    super().__init__()

  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    return batch
