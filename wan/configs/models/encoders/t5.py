from dataclasses import dataclass, field


@dataclass
class T5ArchConfig:
  vocab_size: int = 256384
  dim: int = 4096
  dim_attn: int = 4096
  dim_ffn: int = 10240
  num_heads: int = 64
  num_layers: int = 24
  num_buckets: int = 32
  shared_pos: bool = False
  dropout: float = 0.1

  text_len: int = 512


@dataclass
class T5Config:
  arch_config: T5ArchConfig = field(default_factory=T5ArchConfig)
