import argparse
import statistics

import torch
import torchvision.transforms.functional as TF
from PIL import Image

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
  return parser.parse_args()


def build_video(device):
  img = Image.open(IMAGE_PATH).convert("RGB")
  img = img.resize(IMAGE_SIZE)
  img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device) # [0, 1] -> [-1, 1]
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


def main():
  args = parse_args()
  device = torch.device("cuda:0")
  torch.backends.cudnn.benchmark = True

  vae = Wan2_1_VAE(vae_pth=VAE_PATH, device=device)

  video = build_video(device)

  if args.benchmark:
    warmup(vae, video)
    benchmark(vae, video)
  else:
    compare_encoded(vae, video)


if __name__ == "__main__":
  main()
