import argparse
import statistics

import PIL.Image
import torch

from wan.bench.layer_timer import LayerTimer
from wan.bench.nvtx_marker import NVTXMarker, cuda_profiler_start, cuda_profiler_stop
from wan.configs.pipeline.wan import WanI2VConfig
from wan.configs.sample.wan import Wan2_2_I2V_SamplingParam
from wan.modules.wanvae import Wan2_1_VAE
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.decoding import DecodingStage
from wan.stages.image_encoding import ImageVAEEncodingStage
from wan.stages.schedule_batch import Req
from wan.torch_utils import PRECISION_TO_TYPE, set_default_torch_dtype, skip_init_modules

VAE_PATH = "models/vae/wan_2.1_vae.safetensors"
IMAGE_PATH = "examples/i2v_input.JPG"
ENCODED_PATH = "encoded.pt"
FRAME_NUM = 81
IMAGE_SIZE = (480, 832)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Run the VAE encode benchmark instead of comparing against encoded.pt.",
  )
  parser.add_argument(
    "--profile-layers",
    action="store_true",
    help="Time VAE modules during one encode pass.",
  )
  parser.add_argument(
    "--profile-parents",
    action="store_true",
    help="Include non-leaf parent modules in layer timings. Parent timings include child module time.",
  )
  parser.add_argument(
    "--profile-limit",
    type=int,
    default=40,
    help="Maximum number of timed layers to print.",
  )
  parser.add_argument(
    "--profile-filter",
    type=str,
    default=None,
    help="Only profile modules whose name or class contains this substring.",
  )
  parser.add_argument(
    "--nsys",
    action="store_true",
    help="Run one encode inside cudaProfilerStart/Stop with NVTX module ranges. Launch under `nsys profile --capture-range=cudaProfilerApi`.",
  )
  return parser.parse_args()


def make_batch():
  image = PIL.Image.open(IMAGE_PATH).convert("RGB").resize(IMAGE_SIZE)
  sampling_params = Wan2_2_I2V_SamplingParam(height=IMAGE_SIZE[1], width=IMAGE_SIZE[0], num_frames=FRAME_NUM)
  return Req(sampling_params=sampling_params, condition_image=image)


def encode_once(stage, server_args):
  batch = make_batch()
  with torch.inference_mode():
    stage(batch, server_args)
  return batch.image_latent


def warmup(stage, server_args, run_count=3):
  for _ in range(run_count):
    _ = encode_once(stage, server_args)
  torch.cuda.synchronize()


def compare_encoded(stage, decoding_stage, server_args, device):
  y = encode_once(stage, server_args)
  orig = torch.load(ENCODED_PATH, map_location=device)
  print(orig.shape)

  # encoded.pt was saved before the I2V mask channels were appended; compare only the latent portion.
  y_latent = y[:, : orig.shape[0]]

  decode_batch = Req()
  decode_batch.latents = y_latent
  output_batch = decoding_stage(decode_batch, server_args)
  decoded = output_batch.output  # [B, C, T, H, W] in [0, 1]

  first_frame = decoded[0, :, 0].clamp(0, 1).cpu()
  PIL.Image.fromarray((first_frame.permute(1, 2, 0) * 255).to(torch.uint8).numpy()).save("decoded.png")

  if torch.allclose(orig, y_latent.squeeze(0)):
    print("Original and encoded video are the same")
    return

  max_diff = (orig - y_latent.squeeze(0)).abs().max().item()
  print(f"Original and encoded video are different; max abs diff: {max_diff:.8f}")


def benchmark(stage, server_args):
  times = []
  run_count = 10
  for _ in range(run_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = encode_once(stage, server_args)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)

  print(f"Runs: {run_count}")
  print(f"Min:    {min(times):.6f} sec")
  print(f"Median: {statistics.median(times):.6f} sec")
  print(f"Mean:   {statistics.mean(times):.6f} sec")
  print(f"Std:    {statistics.pstdev(times):.6f} sec")


def profile_nsys(stage, server_args, vae):
  warmup(stage, server_args, run_count=2)
  cuda_profiler_start()
  with NVTXMarker(vae.model):
    torch.cuda.nvtx.range_push("encode")
    _ = encode_once(stage, server_args)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
  cuda_profiler_stop()


def profile_layers(stage, server_args, vae, limit, name_filter, include_parents):
  warmup(stage, server_args, run_count=1)

  with LayerTimer(vae.model, name_filter=name_filter, include_parents=include_parents) as timer:
    _ = encode_once(stage, server_args)

  if include_parents:
    print("Note: parent module timings include child module time.")
  print(f"{'Total ms':>10}  {'Calls':>5}  {'Module':<18}  Name")
  print("-" * 90)
  for total_ms, calls, class_name, name in timer.results_ms()[:limit]:
    print(f"{total_ms:10.3f}  {calls:5d}  {class_name:<18}  {name}")


def main():
  args = parse_args()
  local_torch_device = get_local_torch_device()

  pipeline_config = WanI2VConfig()
  server_args = ServerArgs(pipeline_config=pipeline_config)

  # init vae
  vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]

  with set_default_torch_dtype(vae_dtype), skip_init_modules():
    vae = Wan2_1_VAE(config=pipeline_config.vae_config).to(local_torch_device)

  vae.load(VAE_PATH, server_args)

  # stages
  image_encoding_stage = ImageVAEEncodingStage(vae=vae)
  decoding_stage = DecodingStage(vae=vae)

  if args.nsys:
    profile_nsys(image_encoding_stage, server_args, vae)
  elif args.profile_layers:
    profile_layers(
      image_encoding_stage, server_args, vae, args.profile_limit, args.profile_filter, args.profile_parents
    )
  elif args.benchmark:
    warmup(image_encoding_stage, server_args)
    benchmark(image_encoding_stage, server_args)
  else:
    compare_encoded(image_encoding_stage, decoding_stage, server_args, local_torch_device)


if __name__ == "__main__":
  main()


# Min:    2.414933 sec
# Median: 2.608717 sec
# Mean:   2.600217 sec
# Std:    0.104471 sec
