# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import random
import sys
import warnings
from datetime import datetime

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.configs.wan_i2v_A14B import i2v_A14B
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool

warnings.filterwarnings('ignore')


def _validate_args(args):
  # Basic check
  assert args.ckpt_dir is not None, "Please specify the checkpoint directory."

  cfg = i2v_A14B

  if args.sample_steps is None:
    args.sample_steps = cfg.sample_steps

  if args.sample_shift is None:
    args.sample_shift = cfg.sample_shift

  if args.sample_guide_scale is None:
    args.sample_guide_scale = cfg.sample_guide_scale

  if args.frame_num is None:
    args.frame_num = cfg.frame_num

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
  parser.add_argument(
    "--offload_model", type=str2bool, default=None, help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
  )
  parser.add_argument("--ulysses_size", type=int, default=1, help="The size of the ulysses parallelism in DiT.")
  parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Whether to use FSDP for T5.")
  parser.add_argument("--t5_cpu", action="store_true", default=False, help="Whether to place T5 model on CPU.")
  parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Whether to use FSDP for DiT.")
  parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
  parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
  parser.add_argument("--base_seed", type=int, default=-1, help="The seed to use for generating the video.")
  parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")
  parser.add_argument("--sample_solver", type=str, default='unipc', choices=['unipc', 'dpm++'], help="The solver used to sample.")
  parser.add_argument("--sample_steps", type=int, default=None, help="The sampling steps.")
  parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor for flow matching schedulers.")
  parser.add_argument("--sample_guide_scale", type=float, default=None, help="Classifier free guidance scale.")
  parser.add_argument("--convert_model_dtype", action="store_true", default=False, help="Whether to convert model paramerters dtype.")

  # animate
  parser.add_argument("--src_root_path", type=str, default=None, help="The file of the process output path. Default None.")
  parser.add_argument("--refert_num", type=int, default=77, help="How many frames used for temporal guidance. Recommended to be 1 or 5.")
  parser.add_argument("--replace_flag", action="store_true", default=False, help="Whether to use replace.")
  parser.add_argument("--use_relighting_lora", action="store_true", default=False, help="Whether to use relighting lora.")

  # following args only works for s2v
  parser.add_argument("--num_clip", type=int, default=None, help="Number of video clips to generate, the whole video will not exceed the length of audio.")
  parser.add_argument("--audio", type=str, default=None, help="Path to the audio file, e.g. wav, mp3")
  parser.add_argument("--enable_tts", action="store_true", default=False, help="Use CosyVoice to synthesis audio")
  parser.add_argument(
    "--tts_prompt_audio", type=str, default=None, help="Path to the tts prompt audio file, e.g. wav, mp3. Must be greater than 16khz, and between 5s to 15s."
  )
  parser.add_argument("--tts_prompt_text", type=str, default=None, help="Content to the tts prompt audio. If provided, must exactly match tts_prompt_audio")
  parser.add_argument("--tts_text", type=str, default=None, help="Text wish to synthesize")
  parser.add_argument("--pose_video", type=str, default=None, help="Provide Dw-pose sequence to do Pose Driven")
  parser.add_argument("--start_from_ref", action="store_true", default=False, help="whether set the reference image as the starting point for generation")
  parser.add_argument("--infer_frames", type=int, default=80, help="Number of frames per clip, 48 or 80 or others (must be multiple of 4) for 14B s2v")
  args = parser.parse_args()
  _validate_args(args)

  return args


def _init_logging(rank):
  # logging
  if rank == 0:
    # set format
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", handlers=[logging.StreamHandler(stream=sys.stdout)])
  else:
    logging.basicConfig(level=logging.ERROR)


def generate(args):
  rank = int(os.getenv("RANK", 0))
  world_size = int(os.getenv("WORLD_SIZE", 1))
  local_rank = int(os.getenv("LOCAL_RANK", 0))
  device = local_rank
  _init_logging(rank)

  if args.offload_model is None:
    args.offload_model = False if world_size > 1 else True
    logging.info(f"offload_model is not specified, set to {args.offload_model}.")
  if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
  else:
    assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
    assert not (args.ulysses_size > 1), "sequence parallel are not supported in non-distributed environments."

  if args.ulysses_size > 1:
    assert args.ulysses_size == world_size, "The number of ulysses_size should be equal to the world size."
    init_distributed_group()

  cfg = i2v_A14B
  if args.ulysses_size > 1:
    assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

  logging.info(f"Generation job args: {args}")
  logging.info(f"Generation model config: {cfg}")

  if dist.is_initialized():
    base_seed = [args.base_seed] if rank == 0 else [None]
    dist.broadcast_object_list(base_seed, src=0)
    args.base_seed = base_seed[0]

  logging.info(f"Input prompt: {args.prompt}")
  img = None
  if args.image is not None:
    img = Image.open(args.image).convert("RGB")
    logging.info(f"Input image: {args.image}")

  logging.info("Creating WanI2V pipeline.")
  wan_i2v = wan.WanI2V(
    config=cfg,
    checkpoint_dir=args.ckpt_dir,
    device_id=device,
    rank=rank,
    t5_fsdp=args.t5_fsdp,
    dit_fsdp=args.dit_fsdp,
    use_sp=(args.ulysses_size > 1),
    t5_cpu=args.t5_cpu,
    convert_model_dtype=args.convert_model_dtype,
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
    guide_scale=args.sample_guide_scale,
    seed=args.base_seed,
    offload_model=args.offload_model,
  )

  if rank == 0:
    if args.save_file is None:
      formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
      formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
      suffix = '.mp4'
      args.save_file = (
        f"{args.size.replace('*', 'x') if sys.platform == 'win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}" + suffix
      )

    logging.info(f"Saving generated video to {args.save_file}")
    save_video(tensor=video[None], save_file=args.save_file, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
  del video

  torch.cuda.synchronize()
  if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

  logging.info("Finished.")


if __name__ == "__main__":
  args = _parse_args()
  generate(args)
