import time
from abc import ABC, abstractmethod

import torch

from wan.server_args import ServerArgs, get_global_server_args
from wan.stages.dedup import StageDedupMixin
from wan.stages.schedule_batch import Req


class PipelineStage(StageDedupMixin, ABC):
  def __init__(self):
    self.server_args = get_global_server_args()

  def __call__(self, batch: Req, server_args: ServerArgs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = self.forward(batch, server_args)
    torch.cuda.synchronize()
    print(f"  {type(self).__name__}: {time.perf_counter() - t0:.2f}s")
    return result

  @abstractmethod
  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    raise NotImplementedError
