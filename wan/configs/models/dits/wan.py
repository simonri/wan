from dataclasses import dataclass, field

import torch

from wan.configs.models.dits.base import DiTArchConfig, DiTConfig

_DEFAULT_NEG_PROMPT = (
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
  + "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
  + "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass
class WanArchConfig(DiTArchConfig):
  """Wan DiT architecture parameters (passed to WanModel.__init__)."""

  patch_size: tuple[int, int, int] = (1, 2, 2)
  num_attention_heads: int = 40
  freq_dim: int = 256
  num_layers: int = 40
  eps: float = 1e-6
  qk_norm: str = "rms_norm_across_heads"

  in_dim: int = 36  # 16 noise + 4 mask + 16 encoded-image conditioning
  num_channels_latents: int = 16  # VAE latent channels (== WanModel.out_dim)
  text_dim: int = 4096
  hidden_size: int = 5120
  ffn_dim: int = 13824
  param_dtype: torch.dtype = torch.bfloat16
  boundary_ratio: float = 0.5


@dataclass
class WanConfig(DiTConfig):
  """Wan I2V A14B full config (DiT arch + text encoder + VAE + sampler defaults)."""

  arch_config: WanArchConfig = field(default_factory=WanArchConfig)

  # Text encoder
  t5_model: str = 'umt5_xxl'
  t5_dtype: torch.dtype = torch.bfloat16
  t5_checkpoint: str = 'models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth'
  t5_tokenizer: str = 'google/umt5-xxl'  # HuggingFace id, passed to AutoTokenizer
  text_len: int = 512

  # VAE
  vae_checkpoint: str = 'models/vae/wan_2.1_vae.safetensors'
  vae_stride: tuple[int, int, int] = (4, 8, 8)

  # Diffusion / sampler
  num_train_timesteps: int = 1000
  sample_fps: int = 16
  frame_num: int = 81
  sample_shift: float = 5.0
  sample_steps: int = 8
  sample_guide_scale: tuple[float, float] = (1.0, 1.0)  # (low, high)
  sample_neg_prompt: str = _DEFAULT_NEG_PROMPT
