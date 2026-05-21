from dataclasses import field

from wan.configs.models.base import ArchConfig, ModelConfig


class DiTArchConfig(ArchConfig):
  num_attention_heads: int = 0
  num_channels_latents: int = 0

  hidden_size: int = 0
  boundary_ratio: float | None = None


class DiTConfig(ModelConfig):
  arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)
