from dataclasses import dataclass, field

import torch

from wan.configs.models.dits.base import DiTConfig
from wan.configs.models.encoders.base import TextEncoderConfig
from wan.configs.models.vaes.base import VAEConfig
from wan.stages.schedule_batch import Req


@dataclass
class PipelineConfig:
  # generation params
  flow_shift: float | None = None

  # generation parameters
  generator_device: str | None = None

  # model configuration
  dit_config: DiTConfig = field(default_factory=DiTConfig)
  dit_precision: str = "bf16"

  boundary_ratio: float | None = None

  # text encoder config
  text_encoder_config: TextEncoderConfig = field(default_factory=TextEncoderConfig)
  text_encoder_precision: str = "fp32"

  # vae config
  vae_config: VAEConfig = field(default_factory=VAEConfig)
  vae_precision: str = "fp32"

  def __post_init__(self) -> None:
    """No-op hook so subclasses can safely chain via super()."""

  def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs) -> dict:
    return tokenizer(prompt, **tok_kwargs)

  def get_decode_scale_and_shift(self, device, dtype, vae):
    vae_arch_config = self.vae_config.arch_config
    scaling_factor = getattr(vae_arch_config, "scaling_factor", None)
    if scaling_factor is None:
      scaling_factor = getattr(vae, "scaling_factor", None)

    shift_factor = getattr(vae_arch_config, "shift_factor", None)
    if shift_factor is None:
      shift_factor = getattr(vae, "shift_factor", None)

    return scaling_factor, shift_factor

  def postprocess_image_latent(self, latent_condition: torch.Tensor, batch: Req) -> torch.Tensor:
    vae_arch_config = self.vae_config.arch_config
    spatial_compression_ratio = vae_arch_config.spatial_compression_ratio
    temporal_compression_ratio = vae_arch_config.temporal_compression_ratio
    num_frames = batch.num_frames
    latent_height = batch.height // spatial_compression_ratio
    latent_width = batch.width // spatial_compression_ratio
    mask_lat_size = torch.ones(1, 1, num_frames, latent_height, latent_width)
    mask_lat_size[:, :, 1:] = 0
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, repeats=temporal_compression_ratio, dim=2)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(
      1,
      -1,
      temporal_compression_ratio,
      latent_height,
      latent_width,
    )
    mask_lat_size = mask_lat_size.transpose(1, 2)
    mask_lat_size = mask_lat_size.to(latent_condition.device)
    image_latents = torch.concat([latent_condition, mask_lat_size], dim=1)
    return image_latents

  def prepare_latent_shape(self, batch, batch_size, num_frames):
    height = batch.height // self.vae_config.arch_config.spatial_compression_ratio
    width = batch.width // self.vae_config.arch_config.spatial_compression_ratio

    shape = (
      batch_size,
      self.dit_config.num_channels_latents,
      num_frames,
      height,
      width,
    )

    return shape
