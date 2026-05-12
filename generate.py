# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import random
import sys
import warnings
from datetime import datetime

import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.configs.pipeline.wan import WanI2VConfig
from wan.utils.utils import save_video, str2bool

warnings.filterwarnings('ignore')


def _validate_args(args):
  # Basic check
  assert args.ckpt_dir is not None, "Please specify the checkpoint directory."

  dit_cfg = WanI2VConfig().dit_config

  if args.sample_steps is None:
    args.sample_steps = dit_cfg.sample_steps

  if args.sample_shift is None:
    args.sample_shift = dit_cfg.sample_shift

  if args.sample_guide_scale is None:
    args.sample_guide_scale = dit_cfg.sample_guide_scale

  if args.frame_num is None:
    args.frame_num = dit_cfg.frame_num

  args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
  # Size check
  assert args.size in SUPPORTED_SIZES, (
    f"Unsupport size {args.size}, supported sizes are: {', '.join(SUPPORTED_SIZES)}"
  )


def _parse_args():
  parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
  parser.add_argument(
    "--size",
    type=str,
    default="1280*720",
    choices=list(SIZE_CONFIGS.keys()),
    help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
  )
  parser.add_argument("--frame_num", type=int, default=None, help="How many frames of video are generated. The number should be 4n+1")
  parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory.")
  parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
  parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
  parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the video.")
  parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
  parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample.")
  parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
  parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
  parser.add_argument(
    "--sample_guide_scale",
    type=float,
    nargs='+',
    default=None,
    help="CFG scale. One value (both stages) or two values (low_cfg high_cfg).",
  )
  parser.add_argument(
    "--boundary",
    type=float,
    default=None,
    help="Linear-sigma boundary between low- and high-noise stages (shift-invariant). Overrides config.",
  )
  parser.add_argument(
    "--lora_low",
    nargs='*',
    default=[],
    help="LoRA(s) to merge into the low-noise DiT. Each entry is PATH or PATH:STRENGTH (default 1.0).",
  )
  parser.add_argument(
    "--lora_high",
    nargs='*',
    default=[],
    help="LoRA(s) to merge into the high-noise DiT. Each entry is PATH or PATH:STRENGTH (default 1.0).",
  )

  args = parser.parse_args()
  _validate_args(args)

  return args


def _parse_lora_spec(spec):
  """Parse 'PATH' or 'PATH:STRENGTH' into (path, strength)."""
  if ':' in spec:
    path, _, strength = spec.rpartition(':')
    try:
      return path, float(strength)
    except ValueError:
      pass  # colon was part of the path, not a strength suffix
  return spec, 1.0


def generate(args):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
  )

  cfg = WanI2VConfig()
  logging.info(f"Generation job args: {args}")
  logging.info(f"Generation model config: {cfg}")

  logging.info(f"Input prompt: {args.prompt}")
  img = None
  if args.image is not None:
    img = Image.open(args.image).convert("RGB")
    logging.info(f"Input image: {args.image}")

  low_loras = [_parse_lora_spec(s) for s in args.lora_low]
  high_loras = [_parse_lora_spec(s) for s in args.lora_high]
  for tag, items in [("low", low_loras), ("high", high_loras)]:
    for path, strength in items:
      logging.info(f"LoRA ({tag}, strength={strength}): {path}")

  guide_scale = args.sample_guide_scale
  if guide_scale is not None and len(guide_scale) == 1:
    guide_scale = guide_scale[0]

  logging.info("Creating WanI2V pipeline.")
  wan_i2v = wan.WanI2V(
    config=cfg,
    checkpoint_dir=args.ckpt_dir,
    low_noise_loras=low_loras,
    high_noise_loras=high_loras,
  )
  logging.info("Generating video ...")
  video = wan_i2v.generate(
    args.prompt,
    img,
    max_area=MAX_AREA_CONFIGS[args.size],
    frame_num=args.frame_num,
    shift=args.sample_shift,
    sample_solver=args.sample_solver,
    sampling_steps=args.sample_steps,
    guide_scale=guide_scale,
    boundary=args.boundary,
    seed=args.base_seed,
  )

  if args.save_file is None:
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
    size = args.size.replace('*', 'x') if sys.platform == 'win32' else args.size
    args.save_file = f"{size}_{formatted_prompt}_{formatted_time}.mp4"

  logging.info(f"Saving generated video to {args.save_file}")
  save_video(
    tensor=video[None],
    save_file=args.save_file,
    fps=cfg.dit_config.sample_fps,
    nrow=1,
    normalize=True,
    value_range=(-1, 1),
  )
  del video

  torch.cuda.synchronize()
  logging.info("Finished.")


if __name__ == "__main__":
  args = _parse_args()
  generate(args)
