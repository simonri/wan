from abc import ABC, abstractmethod

from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


class BaseExecutor(ABC):
  def __init__(self, server_args: ServerArgs):
    self.server_args = server_args

  @abstractmethod
  def execute(self, batch: Req) -> Req:
    raise NotImplementedError


class SyncExecutor(BaseExecutor):
  """Synchronously execute a list of pipeline stages."""

  def execute(
    self,
    stages: list[PipelineStage],
    batch: Req,
    server_args: ServerArgs,
  ):
    payload = batch

    for stage in stages:
      payload = stage(payload, server_args)

    return payload
