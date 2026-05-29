import time
from typing import Any

from pydantic import BaseModel, Field


class VideoResponse(BaseModel):
  id: str
  object: str = "video"
  model: str = "sora-2"
  status: str = "queued"
  progress: int = 0
  created_at: int = Field(default_factory=lambda: int(time.time()))
  size: str = ""
  seconds: str = "4"
  quality: str = "standard"
  url: str | None = None
  remixed_from_video_id: str | None = None
  completed_at: int | None = None
  expires_at: int | None = None
  error: dict[str, Any] | None = None
  file_path: str | None = None
  file_paths: list[str] | None = None
  num_outputs: int | None = None
  peak_memory_mb: float | None = None
  inference_time_s: float | None = None


class VideoGenerationsRequest(BaseModel):
  prompt: str
  input_reference: str | None = None
  reference_url: str | None = None
  model: str | None = None
  n: int | None = 1
  num_outputs_per_prompt: int | None = None
  seconds: int | None = 4
  size: str | None = ""
  fps: int | None = None
  num_frames: int | None = None
  seed: int | list[int] | None = None
  generator_device: str | None = "cuda"
  width: int | None = None
  height: int | None = None
  num_inference_steps: int | None = None
  # Frame interpolation
  enable_frame_interpolation: bool | False = False
  frame_interpolation_exp: int | 1 = 1  # 1=2×, 2=4×
  frame_interpolation_scale: float | 1.0 = 1.0
  frame_interpolation_model_path: str | None = None
  # Upscaling
  enable_upscaling: bool | False = False
  upscaling_model_path: str | None = None
  upscaling_scale: int | 4 = 4
  output_quality: str = "default"
  output_compression: int | None = None
  output_path: str | None = None
  diffusers_kwargs: dict[str, Any] | None = None  # kwargs for diffusers backend
  # Performance profiling
  perf_dump_path: str | None = None
