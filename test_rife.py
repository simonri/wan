import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch

from rife import RIFE
from wan.utils.utils import str2bool


def _parse_args():
  parser = argparse.ArgumentParser(description="Interpolate frames in a video using RIFE.")
  parser.add_argument("--input", type=str, default="test_rife.mp4", help="Input video path.")
  parser.add_argument("--output", type=str, default=None, help="Output video path.")
  parser.add_argument("--ckpt_name", type=str, default="rife49", choices=["rife49"], help="RIFE checkpoint.")
  parser.add_argument("--multiplier", type=int, default=2, help="Frame-rate multiplier (>= 2).")
  parser.add_argument("--ensemble", type=str2bool, default=True, help="Run model forward + reversed and average.")
  return parser.parse_args()


def load_video(path):
  reader = imageio.get_reader(path)
  fps = reader.get_meta_data().get("fps", 30.0)
  frames = list(reader.iter_data())
  reader.close()
  return torch.from_numpy(np.stack(frames)).float() / 255.0, fps


def save_video(frames, path, fps):
  frames = (frames.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
  writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
  for frame in frames:
    writer.append_data(frame)
  writer.close()


def main():
  logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
  )

  args = _parse_args()
  logging.info(f"Args: {args}")

  logging.info(f"Loading {args.input}")
  frames, fps = load_video(args.input)
  logging.info(f"Loaded {frames.shape[0]} frames at {fps:.2f} fps, resolution {frames.shape[1]}x{frames.shape[2]}")

  rife = RIFE(ckpt_name=args.ckpt_name)
  out = rife.interpolate(frames, multiplier=args.multiplier, ensemble=args.ensemble)

  if args.output is None:
    stem = Path(args.input).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = f"{stem}_x{args.multiplier}_{timestamp}.mp4"

  out_fps = fps * args.multiplier
  logging.info(f"Saving {out.shape[0]} frames at {out_fps:.2f} fps to {args.output}")
  save_video(out, args.output, out_fps)
  logging.info("Finished.")


if __name__ == "__main__":
  main()
