import torch

from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class DenoisingStage(PipelineStage):
  def __init__(
    self,
    transformer,
    scheduler: FlowUniPCMultistepScheduler,
    pipeline=None,
    transformer_2=None,
    vae=None,
  ):
    super().__init__()
    self.transformer = transformer
    self.transformer_2 = transformer_2

    self.scheduler = scheduler
    self.vae = vae

  @torch.no_grad()
  def forward(self, batch: Req, server_args: ServerArgs) -> Req:
    return batch
