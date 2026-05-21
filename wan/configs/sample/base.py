from dataclasses import dataclass, field


@dataclass
class SamplingParams:
  prompt: str | list[str] = field(default=None)

  # batch info
  num_outputs_per_prompt: int = 1
  seed: int = field(default=42)
  generator_device: str | None = None

  height: int | None = None
  width: int | None = None
  fps: int = 16

  # denoising params
  num_inference_steps: int = None
  guidance_scale: float = 1.0
  guidance_scale_2: float = None
