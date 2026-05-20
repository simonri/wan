from dataclasses import dataclass, field

from wan.configs.models.encoders.base import TextEncoderArchConfig, TextEncoderConfig


@dataclass
class T5ArchConfig(TextEncoderArchConfig):
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

  def __post_init__(self):
    self.tokenizer_kwargs = {
      "padding": "max_length",
      "truncation": True,
      "max_length": self.text_len,
      "return_attention_mask": True,
      "return_tensors": "pt",
    }


@dataclass
class T5Config(TextEncoderConfig):
  arch_config: T5ArchConfig = field(default_factory=T5ArchConfig)
