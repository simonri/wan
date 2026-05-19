from abc import ABC

import torch

from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


class PipelineBase(ABC):
  def __init__(self):
    self._stages: list[PipelineStage] = []

  def add_stage(self, stage: PipelineStage) -> "PipelineBase":
    self._stages.append(stage)
    return self

  @torch.no_grad()
  def forward(self, batch: Req):
    pass
