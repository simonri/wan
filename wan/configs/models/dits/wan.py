from dataclasses import dataclass, field

from wan.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class WanArchConfig(DiTArchConfig):
  param_names_mapping: dict = field(
    default_factory=lambda: {
      r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
      r"^text_embedding\.0\.(.*)$": r"condition_embedder.text_embedder.fc_in.\1",
      r"^text_embedding\.2\.(.*)$": r"condition_embedder.text_embedder.fc_out.\1",
      r"^time_embedding\.0\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_in.\1",
      r"^time_embedding\.2\.(.*)$": r"condition_embedder.time_embedder.mlp.fc_out.\1",
      r"^time_projection\.1\.(.*)$": r"condition_embedder.time_modulation.linear.\1",
      r"^head\.head\.(.*)$": r"proj_out.\1",
      r"^head\.modulation$": r"scale_shift_table",
      r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.to_q.\2",
      r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.to_k.\2",
      r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.to_v.\2",
      r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.to_out.\2",
      r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
      r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
      r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
      r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
      r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
      r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
      r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": r"blocks.\1.attn2.norm_q.\2",
      r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": r"blocks.\1.attn2.norm_k.\2",
      r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
      r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
      r"^blocks\.(\d+)\.norm3\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
      r"^blocks\.(\d+)\.modulation$": r"blocks.\1.scale_shift_table",
    }
  )

  # [._] handles both diffusers (blocks.0.self_attn.q.…) and kohya (blocks_0_self_attn_q.…) style.
  lora_param_names_mapping: dict = field(
    default_factory=lambda: {
      r"^blocks[._](\d+)[._]self_attn[._]q\.(.*)$": r"blocks.\1.to_q.\2",
      r"^blocks[._](\d+)[._]self_attn[._]k\.(.*)$": r"blocks.\1.to_k.\2",
      r"^blocks[._](\d+)[._]self_attn[._]v\.(.*)$": r"blocks.\1.to_v.\2",
      r"^blocks[._](\d+)[._]self_attn[._]o\.(.*)$": r"blocks.\1.to_out.\2",
      r"^blocks[._](\d+)[._]cross_attn[._]q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
      r"^blocks[._](\d+)[._]cross_attn[._]k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
      r"^blocks[._](\d+)[._]cross_attn[._]v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
      r"^blocks[._](\d+)[._]cross_attn[._]o\.(.*)$": r"blocks.\1.attn2.to_out.\2",
      r"^blocks[._](\d+)[._]ffn[._]0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
      r"^blocks[._](\d+)[._]ffn[._]2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
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
