import argparse
import statistics

import torch

from rife.rife import RIFE, SCALE_LIST


def parse_args():
  parser = argparse.ArgumentParser(description="Benchmark RIFE frame-interpolation speed.")
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument(
    "--multiplier", type=int, default=2, help=">=2; produces multiplier-1 intermediate frames per pair."
  )
  parser.add_argument("--ensemble", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--ckpt", type=str, default="rife49")
  parser.add_argument("--runs", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=5)
  return parser.parse_args()


def build_pair(h, w, device, seed=0):
  """Random pair of frames, (1, 3, H, W) in [0, 1]."""
  g = torch.Generator(device=device).manual_seed(seed)
  img0 = torch.rand(1, 3, h, w, generator=g, device=device)
  img1 = torch.rand(1, 3, h, w, generator=g, device=device)
  return img0, img1


def run_pair(model, img0, img1, ensemble, multiplier):
  """One pair's worth of work: (multiplier - 1) model calls."""
  with torch.no_grad():
    for j in range(1, multiplier):
      _ = model(img0, img1, timestep=j / multiplier, scale_list=SCALE_LIST, ensemble=ensemble)


def warmup(model, img0, img1, ensemble, multiplier, runs):
  for _ in range(runs):
    run_pair(model, img0, img1, ensemble, multiplier)
  torch.cuda.synchronize()


def benchmark(model, img0, img1, ensemble, multiplier, runs):
  times = []
  for _ in range(runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run_pair(model, img0, img1, ensemble, multiplier)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)
  return times


def print_stats(times, args):
  intermediate_per_pair = args.multiplier - 1
  median = statistics.median(times)
  print(f"Resolution:   {args.height}x{args.width}")
  print(f"Multiplier:   {args.multiplier} ({intermediate_per_pair} intermediate frame(s) per pair)")
  print(f"Ensemble:     {args.ensemble}")
  print(f"Runs:         {args.runs} (after {args.warmup} warmup)")
  print(f"Min:          {min(times) * 1000:.2f} ms / pair")
  print(f"Median:       {median * 1000:.2f} ms / pair")
  print(f"Mean:         {statistics.mean(times) * 1000:.2f} ms / pair")
  print(f"Std:          {statistics.pstdev(times) * 1000:.2f} ms")
  print(f"Throughput:   {1.0 / median:.2f} pairs / sec  ({intermediate_per_pair / median:.2f} intermediate fps)")


def main():
  args = parse_args()
  device = torch.device("cuda:0")
  torch.backends.cudnn.benchmark = True

  rife = RIFE(ckpt_name=args.ckpt, device=device)
  img0, img1 = build_pair(args.height, args.width, device)

  warmup(rife.model, img0, img1, args.ensemble, args.multiplier, args.warmup)
  times = benchmark(rife.model, img0, img1, args.ensemble, args.multiplier, args.runs)
  print_stats(times, args)


if __name__ == "__main__":
  main()
