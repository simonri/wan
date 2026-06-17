from dataclasses import dataclass, field

from wan.configs.models.base import ArchConfig, ModelConfig
from wan.layers.quantization.config.base_config import QuantizationConfig


@dataclass
class DiTArchConfig(ArchConfig):
  param_names_mapping: dict = field(default_factory=dict)
  lora_param_names_mapping: dict = field(default_factory=dict)

  num_attention_heads: int = 0
  num_channels_latents: int = 0

  hidden_size: int = 0
  boundary_ratio: float | None = None


@dataclass
class DiTConfig(ModelConfig):
  arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)
  quant_config: QuantizationConfig | None = None
