from dataclasses import dataclass, field

import torch

from wan.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class WanArchConfig(DiTArchConfig):
  param_names_mapping: dict = field(
    default_factory=lambda: {
      "patch_embedding.weight": "patch_embedding.proj.weight",
      "patch_embedding.bias": "patch_embedding.proj.bias",
      "text_embedding.0.weight": "condition_embedder.text_embedder.fc_in.weight",
      "text_embedding.0.bias": "condition_embedder.text_embedder.fc_in.bias",
      "text_embedding.2.weight": "condition_embedder.text_embedder.fc_out.weight",
      "text_embedding.2.bias": "condition_embedder.text_embedder.fc_out.bias",
      "time_embedding.0.weight": "condition_embedder.time_embedder.mlp.fc_in.weight",
      "time_embedding.0.bias": "condition_embedder.time_embedder.mlp.fc_in.bias",
      "time_embedding.2.weight": "condition_embedder.time_embedder.mlp.fc_out.weight",
      "time_embedding.2.bias": "condition_embedder.time_embedder.mlp.fc_out.bias",
      "time_projection.1.weight": "condition_embedder.time_modulation.linear.weight",
      "time_projection.1.bias": "condition_embedder.time_modulation.linear.bias",
      "head.head.weight": "proj_out.weight",
      "head.head.bias": "proj_out.bias",
      "head.modulation": "scale_shift_table",
    }
  )

  block_param_names_mapping: dict = field(
    default_factory=lambda: {
      "self_attn.q.weight": "to_q.weight",
      "self_attn.q.bias": "to_q.bias",
      "self_attn.k.weight": "to_k.weight",
      "self_attn.k.bias": "to_k.bias",
      "self_attn.v.weight": "to_v.weight",
      "self_attn.v.bias": "to_v.bias",
      "self_attn.o.weight": "to_out.weight",
      "self_attn.o.bias": "to_out.bias",
      "self_attn.norm_q.weight": "norm_q.weight",
      "self_attn.norm_k.weight": "norm_k.weight",
      "modulation": "scale_shift_table",
      "ffn.0.weight": "ffn.fc_in.weight",
      "ffn.0.bias": "ffn.fc_in.bias",
      "ffn.2.weight": "ffn.fc_out.weight",
      "ffn.2.bias": "ffn.fc_out.bias",
      "norm3.weight": "self_attn_residual_norm.norm.weight",
      "norm3.bias": "self_attn_residual_norm.norm.bias",
      "cross_attn.q.weight": "attn2.to_q.weight",
      "cross_attn.q.bias": "attn2.to_q.bias",
      "cross_attn.k.weight": "attn2.to_k.weight",
      "cross_attn.k.bias": "attn2.to_k.bias",
      "cross_attn.v.weight": "attn2.to_v.weight",
      "cross_attn.v.bias": "attn2.to_v.bias",
      "cross_attn.o.weight": "attn2.to_out.weight",
      "cross_attn.o.bias": "attn2.to_out.bias",
      "cross_attn.norm_q.weight": "attn2.norm_q.weight",
      "cross_attn.norm_k.weight": "attn2.norm_k.weight",
    }
  )

  lora_param_names_mapping: dict = field(
    default_factory=lambda: {
      "self_attn.q": "to_q",
      "self_attn.k": "to_k",
      "self_attn.v": "to_v",
      "self_attn.o": "to_out",
      "cross_attn.q": "attn2.to_q",
      "cross_attn.k": "attn2.to_k",
      "cross_attn.v": "attn2.to_v",
      "cross_attn.o": "attn2.to_out",
      "ffn.0": "ffn.fc_in",
      "ffn.2": "ffn.fc_out",
    }
  )

  patch_size: tuple[int, int, int] = (1, 2, 2)
  num_attention_heads: int = 40
  freq_dim: int = 256
  num_layers: int = 40
  eps: float = 1e-6
  qk_norm: str = "rms_norm_across_heads"

  attention_head_dim: int = 128

  in_dim: int = 36  # 16 noise + 4 mask + 16 encoded-image conditioning
  num_channels_latents: int = 16  # VAE latent channels (== WanModel.out_dim)
  text_dim: int = 4096
  hidden_size: int = 5120
  ffn_dim: int = 13824

  # wan moe
  boundary_ratio: float | None = None

  def __post_init__(self):
    self.hidden_size = self.num_attention_heads * self.attention_head_dim


@dataclass
class WanConfig(DiTConfig):
  """Wan I2V A14B full config (DiT arch + text encoder + VAE + sampler defaults)."""

  arch_config: WanArchConfig = field(default_factory=WanArchConfig)
