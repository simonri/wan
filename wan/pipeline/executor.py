import contextlib
from abc import ABC, abstractmethod
from collections.abc import Callable

from wan.profiler import DiffusionProfiler
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

  @contextlib.contextmanager
  def profile_execution(self, batch: Req):
    do_profile = batch.profile
    full_profile = batch.profile_all_stages
    if not do_profile:
      yield
      return

    profiler = DiffusionProfiler(
      full_profile=full_profile,
      num_steps=batch.num_profiled_timesteps,
      num_inference_steps=batch.num_inference_steps,
    )
    try:
      yield
    finally:
      profiler.stop()

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
    with self.profile_execution(batch):
      batch = self._run_all_stages(
        stages, batch, server_args, lambda stage, payload, server_args: stage(payload, server_args)
      )

    return batch

  def execute_group(
    self, stages: list[PipelineStage], batches: list[Req], server_args: ServerArgs
  ) -> list[OutputBatch]:
    with self.profile_execution(batches[0]):
      batches = self._run_all_stages(
        stages,
        batches,
        server_args,
        lambda stage, batches, server_args: stage.run_grouped_requests(batches, server_args),
      )

    return batches
