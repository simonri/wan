from abc import ABC, abstractmethod

from wan.server_args import ServerArgs
from wan.stages.schedule_batch import Req


class PipelineStage(ABC):
  def __init__(self):
    pass

  def __call__(self, batch: Req, server_args: ServerArgs):
    result = self.forward(batch, server_args)

    return result

  @abstractmethod
  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    raise NotImplementedError
