from dataclasses import dataclass, field


@dataclass
class ArchConfig:
  pass


@dataclass
class ModelConfig:
  # Every model config parameter can be categorized into either ArchConfig or everything else
  arch_config: ArchConfig = field(default_factory=ArchConfig)
