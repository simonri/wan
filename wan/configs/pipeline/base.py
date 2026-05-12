from dataclasses import dataclass, field

from wan.configs.models.dits.base import DiTConfig


@dataclass
class PipelineConfig:
  # generation params
  flow_shift: float | None = None

  # model configuration
  dit_config: DiTConfig = field(default_factory=DiTConfig)
  dit_precision: str = "bf16"

  boundary_ratio: float | None = None

  def __post_init__(self) -> None:
    """No-op hook so subclasses can safely chain via super()."""
