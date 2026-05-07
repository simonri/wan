import argparse
import statistics

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .bench.layer_timer import LayerTimer
from .modules.vae2_1 import Wan2_1_VAE
from .utils.utils import save_video

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
  return parser.parse_args()


def build_video(device):
  img = Image.open(IMAGE_PATH).convert("RGB")
  img = img.resize(IMAGE_SIZE)
  img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)  # [0, 1] -> [-1, 1]
  h, w = img.shape[1:]

  print(f"Image size: {h}x{w}")

  return torch.cat(
    [
      img[:, None],
      torch.zeros(3, FRAME_NUM - 1, h, w, device=device, dtype=img.dtype),
    ],
    dim=1,
  )


def encode_once(vae, video):
  with torch.inference_mode():
    y = vae.encode([video])[0]
  return y


def decode_once(vae, z):
  with torch.inference_mode():
    y = vae.decode([z])[0]
  return y


def warmup(vae, video, run_count=3):
  for _ in range(run_count):
    _ = encode_once(vae, video)
  torch.cuda.synchronize()


def compare_encoded(vae, video):
  y = encode_once(vae, video)
  orig = torch.load(ENCODED_PATH, map_location=video.device)
  print(orig.shape)

  decoded = decode_once(vae, orig)

  save_video(tensor=decoded.unsqueeze(0), save_file="decoded.mp4", fps=16, nrow=1, normalize=True, value_range=(-1, 1))

  if torch.allclose(orig, y):
    print("Original and encoded video are the same")
    return

  max_diff = (orig - y).abs().max().item()
  print(f"Original and encoded video are different; max abs diff: {max_diff:.8f}")


def benchmark(vae, video):
  times = []
  run_count = 10
  for _ in range(run_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = encode_once(vae, video)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)

  print(f"Runs: {run_count}")
  print(f"Min:    {min(times):.6f} sec")
  print(f"Median: {statistics.median(times):.6f} sec")
  print(f"Mean:   {statistics.mean(times):.6f} sec")
  print(f"Std:    {statistics.pstdev(times):.6f} sec")


def profile_layers(vae, video, limit, name_filter, include_parents):
  warmup(vae, video, run_count=1)

  with torch.inference_mode(), LayerTimer(vae.model, name_filter=name_filter, include_parents=include_parents) as timer:
    _ = vae.encode([video])[0]

  if include_parents:
    print("Note: parent module timings include child module time.")
  print(f"{'Total ms':>10}  {'Calls':>5}  {'Module':<18}  Name")
  print("-" * 90)
  for total_ms, calls, class_name, name in timer.results_ms()[:limit]:
    print(f"{total_ms:10.3f}  {calls:5d}  {class_name:<18}  {name}")


def main():
  args = parse_args()
  device = torch.device("cuda:0")
  torch.backends.cudnn.benchmark = True

  vae = Wan2_1_VAE(vae_pth=VAE_PATH, device=device)

  video = build_video(device)

  if args.profile_layers:
    profile_layers(vae, video, args.profile_limit, args.profile_filter, args.profile_parents)
  elif args.benchmark:
    warmup(vae, video)
    benchmark(vae, video)
  else:
    compare_encoded(vae, video)


if __name__ == "__main__":
  main()

#   Total ms  Calls  Module              Name
# ------------------------------------------------------------------------------------------
#   2416.949     41  Encoder3d           encoder
#    625.929     41  ResidualBlock       encoder.downsamples.0
#    624.698     41  ResidualBlock       encoder.downsamples.1
#    356.304     41  ResidualBlock       encoder.downsamples.4
#    301.295     41  ResidualBlock       encoder.downsamples.3
#    208.585     41  CausalConv3d        encoder.downsamples.0.residual.2
#    208.352     41  CausalConv3d        encoder.downsamples.0.residual.6
#    208.211     41  CausalConv3d        encoder.downsamples.1.residual.2
#    207.950     41  CausalConv3d        encoder.downsamples.1.residual.6
#    125.175     41  CausalConv3d        encoder.downsamples.3.residual.6
#    125.065     41  CausalConv3d        encoder.downsamples.4.residual.6
#    125.002     41  CausalConv3d        encoder.downsamples.4.residual.2
#    119.861     21  ResidualBlock       encoder.downsamples.7
#     97.824     21  ResidualBlock       encoder.downsamples.6
#     71.284     41  CausalConv3d        encoder.downsamples.3.residual.2
#     66.838     41  RMS_norm            encoder.downsamples.0.residual.0
#     66.719     41  RMS_norm            encoder.downsamples.0.residual.3
#     66.616     41  RMS_norm            encoder.downsamples.1.residual.3
#     66.566     41  RMS_norm            encoder.downsamples.1.residual.0
#     55.530     41  Resample            encoder.downsamples.2
#     55.521     11  AttentionBlock      encoder.middle.1
#     55.414     41  Sequential          encoder.downsamples.2.resample
#     50.401     41  CausalConv3d        encoder.conv1
#     46.069     21  CausalConv3d        encoder.downsamples.7.residual.6
#     46.041     21  CausalConv3d        encoder.downsamples.6.residual.6
#     46.013     21  CausalConv3d        encoder.downsamples.7.residual.2
#     44.057     41  Resample            encoder.downsamples.5
#     33.804     41  RMS_norm            encoder.downsamples.4.residual.0
#     33.788     41  RMS_norm            encoder.downsamples.3.residual.3
#     33.767     41  RMS_norm            encoder.downsamples.4.residual.3
#     32.966     41  Conv2d              encoder.downsamples.2.resample.1
#     30.118     41  Sequential          encoder.downsamples.5.resample
#     25.480     21  CausalConv3d        encoder.downsamples.6.residual.2
#     22.271     41  ZeroPad2d           encoder.downsamples.2.resample.0
#     20.845     41  CausalConv3d        encoder.downsamples.3.shortcut
#     18.344     41  RMS_norm            encoder.downsamples.3.residual.0
#     18.279     41  Conv2d              encoder.downsamples.5.resample.1
#     17.622     11  ResidualBlock       encoder.downsamples.9
#     17.134     11  ResidualBlock       encoder.downsamples.10
#     17.067     11  ResidualBlock       encoder.middle.0
