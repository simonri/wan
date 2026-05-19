import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import imageio

from rife import RIFE


def _parse_args():
  parser = argparse.ArgumentParser(description="Interpolate frames in a video using RIFE.")
  parser.add_argument("--input", type=str, default="test_rife.mp4", help="Input video path.")
  parser.add_argument("--output", type=str, default=None, help="Output video path.")
  parser.add_argument("--ckpt_name", type=str, default="flownet", choices=["flownet"], help="RIFE checkpoint.")
  return parser.parse_args()


def load_video(path):
  """Return (list of uint8 HxWx3 frames, fps)."""
  reader = imageio.get_reader(path)
  fps = reader.get_meta_data().get("fps", 30.0)
  frames = list(reader.iter_data())
  reader.close()
  return frames, fps


def save_video(frames, path, fps):
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
  h, w = frames[0].shape[:2]
  logging.info(f"Loaded {len(frames)} frames at {fps:.2f} fps, resolution {h}x{w}")

  rife = RIFE(ckpt_name=args.ckpt_name)
  out, multiplier = rife.interpolate(frames, exp=1)

  if args.output is None:
    stem = Path(args.input).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = f"{stem}_{timestamp}.mp4"

  out_fps = fps * multiplier
  logging.info(f"Saving {len(out)} frames at {out_fps:.2f} fps to {args.output}")
  save_video(out, args.output, out_fps)
  logging.info("Finished.")


if __name__ == "__main__":
  main()
