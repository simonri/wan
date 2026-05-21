import argparse
import logging
import sys

import torch

from wan.configs.pipeline.wan import WanI2VConfig
from wan.configs.sample.wan import Wan2_2_I2V_SamplingParam
from wan.pipeline.executor import SyncExecutor
from wan.pipeline.wan_i2v_pipeline import WanImageToVideoPipeline
from wan.server_args import ServerArgs
from wan.stages.schedule_batch import Req


def _parse_args():
  parser = argparse.ArgumentParser(description="Generate a image or video from a text prompt or image using Wan")
  parser.add_argument("--save_file", type=str, default=None, help="The file to save the generated video to.")
  parser.add_argument("--prompt", type=str, default=None, help="The prompt to generate the video from.")
  parser.add_argument("--image", type=str, default=None, help="The image to generate the video from.")

  args = parser.parse_args()
  return args


def generate(args):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
  )

  cfg = WanI2VConfig()
  server_args = ServerArgs(pipeline_config=cfg)

  logging.info("Creating WanI2V pipeline.")
  executor = SyncExecutor(server_args=server_args)

  wan_i2v = WanImageToVideoPipeline(
    server_args=server_args,
    executor=executor,
  )

  sampling_params = Wan2_2_I2V_SamplingParam(height=832, width=480, num_frames=81, num_inference_steps=8)

  req = Req(
    sampling_params=sampling_params,
  )

  req.prompt = args.prompt
  req.image_path = args.image

  wan_i2v.forward(
    req,
    server_args,
  )

  # if args.save_file is None:
  #   formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  #   formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
  #   size = args.size.replace('*', 'x') if sys.platform == 'win32' else args.size
  #   args.save_file = f"{size}_{formatted_prompt}_{formatted_time}.mp4"

  # logging.info(f"Saving generated video to {args.save_file}")
  # save_video(
  #   tensor=video[None],
  #   save_file=args.save_file,
  #   fps=cfg.dit_config.sample_fps,
  #   nrow=1,
  #   normalize=True,
  #   value_range=(-1, 1),
  # )
  # del video

  torch.cuda.synchronize()
  logging.info("Finished.")


if __name__ == "__main__":
  args = _parse_args()
  generate(args)
