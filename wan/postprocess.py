import os
from collections.abc import Callable, Sequence
from typing import Any

import imageio
import torch

from rife.rife import interpolate_video_frames


def post_process_sample(
  sample: torch.Tensor,
  fps: int,
  save_output: bool,
  save_file_path: str,
  output_compression: int | None = None,
  enable_frame_interpolation: bool = False,
  frame_interpolation_exp: int = 1,
  frame_interpolation_scale: float = 1.0,
):
  frames = None

  # 1. convert tensor to list of uin8 HWC frames
  if sample.dim() == 3:
    sample = sample.unsqueeze(1)
  sample = (sample * 255).clamp(0, 255).to(torch.uint8)
  videos = sample.permute(1, 2, 3, 0).cpu().numpy()
  frames = list(videos)

  # 2. frame interpolation
  if enable_frame_interpolation and len(frames) > 1:
    frames, multiplier = interpolate_video_frames(
      frames,
      exp=frame_interpolation_exp,
      scale=frame_interpolation_scale,
    )
    fps = fps * multiplier

  # 4. save outputs if requested
  if save_output:
    if save_file_path:
      os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
      quality = output_compression / 10 if output_compression is not None else 5
      imageio.mimsave(
        save_file_path,
        frames,
        fps=fps,
        format="mp4",
        codec="libx264",
        quality=quality,
      )
    else:
      print("No output path provided, skipping save.")

  return frames


def save_outputs(
  outputs: Sequence[torch.Tensor],
  fps: int,
  save_output: bool,
  build_output_path: Callable[[int], str],
  *,
  output_compression: int | None = None,
  frames_out: list[Any] | None = None,
  enable_frame_interpolation: bool = False,
  frame_interpolation_exp: int = 1,
  frame_interpolation_scale: float = 1.0,
) -> list[str]:
  output_paths: list[str] = []

  for idx, output in enumerate(outputs):
    save_file_path = build_output_path(idx)
    sample = output

    frames = post_process_sample(
      sample,
      fps,
      save_output,
      save_file_path,
      output_compression=output_compression,
      enable_frame_interpolation=enable_frame_interpolation,
      frame_interpolation_exp=frame_interpolation_exp,
      frame_interpolation_scale=frame_interpolation_scale,
    )

    if frames_out is not None:
      frames_out.append(frames)

    output_paths.append(save_file_path)

  return output_paths
