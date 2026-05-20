from dataclasses import dataclass, field

from wan.configs.sample.base import SamplingParams


@dataclass
class Wan2_2_I2V_SamplingParam(SamplingParams):
  negative_prompt: str | None = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多§，倒着走"
  )

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
