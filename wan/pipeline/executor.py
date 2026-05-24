from abc import ABC, abstractmethod
from collections.abc import Callable

from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import OutputBatch, Req


class BaseExecutor(ABC):
  def __init__(self, server_args: ServerArgs):
    self.server_args = server_args

  @abstractmethod
  def execute(self, batch: Req) -> OutputBatch:
    raise NotImplementedError


class SyncExecutor(BaseExecutor):
  """Synchronously execute a list of pipeline stages."""

  def _run_all_stages(
    self,
    stages: list[PipelineStage],
    payload: any,
    server_args: ServerArgs,
    run_stage: Callable[[PipelineStage, any, ServerArgs], any],
  ) -> any:
    for stage in stages:
      payload = run_stage(stage, payload, server_args)

    return payload

  def execute(
    self,
    stages: list[PipelineStage],
    batch: Req,
    server_args: ServerArgs,
  ) -> OutputBatch:
    return self._run_all_stages(
      stages, batch, server_args, lambda stage, payload, server_args: stage(payload, server_args)
    )

  def execute_group(
    self, stages: list[PipelineStage], batches: list[Req], server_args: ServerArgs
  ) -> list[OutputBatch]:
    return self._run_all_stages(
      stages, batches, server_args, lambda stage, batches, server_args: stage.run_grouped_requests(batches, server_args)
    )
