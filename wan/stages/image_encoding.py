from wan.modules.vae2_1 import Wan2_1_VAE
from wan.stages.base import PipelineStage


class ImageVAEEncodingStage(PipelineStage):
  def __init__(self, vae: Wan2_1_VAE, **kwargs):
    super().__init__()
    self.vae = vae

  def forward(self):
    pass
