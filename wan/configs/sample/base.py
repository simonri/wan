from dataclasses import dataclass


@dataclass
class SamplingParams:
  height: int | None = None
  width: int | None = None
  fps: int = 16

  # denoising params
  num_inference_steps: int = None
  guidance_scale: float = 1.0
  guidance_scale_2: float = None
