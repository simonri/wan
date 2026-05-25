from dataclasses import dataclass, field


@dataclass
class SamplingParams:
  prompt: str | list[str] = field(default=None)

  output_path: str | None = None
  output_file_name: str | None = None

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
  boundary_ratio: float | None = None

  # profiling
  profile: bool = False
  num_profiled_timesteps: int = 2
  profile_all_stages: bool = False

  @property
  def n_tokens(self) -> int:
    if self.height and self.width:
      latents_size = [
        (self.num_frames - 1) // 4 + 1,
        self.height // 8,
        self.width // 8,
      ]
      n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
    else:
      n_tokens = -1
    return n_tokens
