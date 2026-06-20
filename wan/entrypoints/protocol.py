import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class LoRAItem(BaseModel):
  """One LoRA to apply to the request. Matches the per-adapter args expected by
  LoRAPipeline.set_lora (nickname, path, target transformer, strength)."""

  nickname: str
  path: str
  target: Literal["transformer", "transformer_2", "all"] = "transformer"
  strength: float = 1.0


class VideoResponse(BaseModel):
  id: str
  status: str = "queued"
  progress: int = 0
  created_at: int = Field(default_factory=lambda: int(time.time()))
  completed_at: int | None = None
  error: dict[str, Any] | None = None
  file_path: str | None = None
  file_paths: list[str] | None = None
  num_outputs: int | None = None
  peak_memory_mb: float | None = None
  inference_time_s: float | None = None


class VideoGenerationsRequest(BaseModel):
  prompt: str
  input_reference: str | None = None
  end_image: str | None = None
  num_outputs_per_prompt: int = 1
  fps: int = 16
  num_frames: int = 81
  seed: int | list[int] | None = None
  generator_device: str = "cuda"
  width: int = 720
  height: int = 1280
  num_inference_steps: int = 8
  # Frame interpolation
  enable_frame_interpolation: bool = False
  frame_interpolation_exp: int = 1  # 1=2×, 2=4×
  frame_interpolation_scale: float = 1.0
  # LoRAs to apply on this request. Stacked in order onto their target transformer
  # (matching LoRAPipeline.set_lora semantics: first LoRA per target clears, the
  # rest add on top). The pipeline caches by nickname, so repeated requests with
  # the same set hit a fast path that just re-applies merged weights.
  loras: list[LoRAItem] | None = None
