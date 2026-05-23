from dataclasses import dataclass, field

import torch

from wan.configs.models.dits.base import DiTConfig
from wan.configs.models.dits.wan import WanConfig
from wan.configs.models.encoders.base import TextEncoderConfig
from wan.configs.models.encoders.t5 import T5Config
from wan.configs.models.vaes.base import VAEConfig
from wan.configs.models.vaes.wanvae import WanVAEConfig
from wan.configs.pipeline.base import BaseEncoderOutput, PipelineConfig


@dataclass
class WanI2VConfig(PipelineConfig):
  dit_config: DiTConfig = field(default_factory=WanConfig)
  max_area: int = 720 * 1280
  flow_shift: float | None = 5.0
  boundary_ratio: float | None = 0.900

  # precision for each component
  precision: str = "bf16"
  vae_precision: str = "fp32"
  text_encoder_precision: str = "bf16"

  # text encoding stage
  text_encoder_config: TextEncoderConfig = field(default_factory=T5Config)

  # vae
  vae_config: VAEConfig = field(default_factory=WanVAEConfig)

  def __post_init__(self) -> None:
    super().__post_init__()
    self.dit_config.boundary_ratio = self.boundary_ratio

  def postprocess_text(self, outputs: BaseEncoderOutput, text_inputs: dict) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_masks
    hidden_states: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_states).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_states, seq_lens, strict=True)]
    return torch.stack(
      [
        torch.cat([u, u.new_zeros(self.text_encoder_config.arch_config.text_len - u.size(0), u.size(1))])
        for u in prompt_embeds
      ],
      dim=0,
    )
