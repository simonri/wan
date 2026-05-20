from dataclasses import dataclass, field

import torch

from wan.configs.models.base import ArchConfig, ModelConfig


@dataclass
class VAEArchConfig(ArchConfig):
  scaling_factor: float | torch.Tensor = 0
  temporal_compression_ratio: int = 4
  spatial_compression_ratio: int = 8


@dataclass
class VAEConfig(ModelConfig):
  arch_config: VAEArchConfig = field(default_factory=VAEArchConfig)
