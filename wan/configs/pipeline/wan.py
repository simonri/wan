from dataclasses import dataclass, field

from wan.configs.models.dits.base import DiTConfig
from wan.configs.models.dits.wan import WanConfig
from wan.configs.pipeline.base import PipelineConfig


@dataclass
class WanI2VConfig(PipelineConfig):
  dit_config: DiTConfig = field(default_factory=WanConfig)
  max_area: int = 720 * 1280
  flow_shift: float | None = 5.0
  boundary_ratio: float | None = 0.900

  precision: str = "bf16"

  def __post_init__(self) -> None:
    super().__post_init__()
    self.dit_config.boundary_ratio = self.boundary_ratio
