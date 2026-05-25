from dataclasses import dataclass, field

from wan.configs.sample.base import SamplingParams


@dataclass
class Wan2_2_I2V_SamplingParam(SamplingParams):
  guidance_scale: float = 1.0
  guidance_scale_2: float = 1.0
  num_inference_steps: int = 8
  fps: int = 16

  num_frames: int = 81

  supported_resolutions: list[tuple[int, int]] | None = field(
    default_factory=lambda: [
      (1280, 720),
      (720, 1280),
      (832, 480),
      (480, 832),
    ]
  )
