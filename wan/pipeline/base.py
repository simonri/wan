import os
from abc import ABC

import torch

from wan.pipeline.executor import BaseExecutor
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import OutputBatch, Req

# avoid deadlocks when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PipelineBase(ABC):
  def __init__(self, executor: BaseExecutor):
    self._stages: list[PipelineStage] = []
    self.executor = executor

  def add_stage(self, stage: PipelineStage) -> "PipelineBase":
    self._stages.append(stage)
    return self

  @property
  def stages(self) -> list[PipelineStage]:
    return self._stages

  @torch.no_grad()
  def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
    print("Running pipeline stages")
    output_batch = self.executor.execute(self._stages, batch, server_args)
    return output_batch
