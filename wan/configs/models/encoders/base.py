from dataclasses import dataclass, field

from wan.configs.models.base import ArchConfig, ModelConfig


@dataclass
class TextEncoderArchConfig(ArchConfig):
  text_len: int = 0

  def __post_init__(self) -> None:
    self.tokenizer_kwargs = {
      "truncation": True,
      "max_length": self.text_len,
      "return_tensors": "pt",
    }


@dataclass
class TextEncoderConfig(ModelConfig):
  arch_config: ArchConfig = field(default_factory=ArchConfig)
