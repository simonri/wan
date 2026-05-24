from abc import ABC, abstractmethod

from wan.server_args import ServerArgs, get_global_server_args
from wan.stages.dedup import StageDedupMixing
from wan.stages.schedule_batch import Req


class PipelineStage(StageDedupMixing, ABC):
  def __init__(self):
    self.server_args = get_global_server_args()

  def __call__(self, batch: Req, server_args: ServerArgs):
    result = self.forward(batch, server_args)

    return result

  @abstractmethod
  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    raise NotImplementedError
