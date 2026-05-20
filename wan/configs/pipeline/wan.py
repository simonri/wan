from dataclasses import dataclass, field

from wan.configs.models.dits.base import DiTConfig
from wan.configs.models.dits.wan import WanConfig
from wan.configs.models.encoders.base import TextEncoderConfig
from wan.configs.models.encoders.t5 import T5Config
from wan.configs.models.vaes.base import VAEConfig
from wan.configs.models.vaes.wanvae import WanVAEConfig
from wan.configs.pipeline.base import PipelineConfig


@dataclass
class WanI2VConfig(PipelineConfig):
  dit_config: DiTConfig = field(default_factory=WanConfig)
  max_area: int = 720 * 1280
  flow_shift: float | None = 5.0
  boundary_ratio: float | None = 0.900

  precision: str = "bf16"

  # text encoding stage
  text_encoder_config: TextEncoderConfig = field(default_factory=T5Config)
  text_encoder_precision: str = "fp32"

  # vae
  vae_config: VAEConfig = field(default_factory=WanVAEConfig)

  def __post_init__(self) -> None:
    super().__post_init__()
    self.dit_config.boundary_ratio = self.boundary_ratio
