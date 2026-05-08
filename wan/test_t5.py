import argparse
import statistics

import torch

from .modules.t5 import T5EncoderModel

CHECKPOINT_PATH = "models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_PATH = "google/umt5-xxl"
TEXT_LEN = 512
DTYPE = torch.bfloat16
DEFAULT_PROMPT = "A cat playing piano in a cozy living room, cinematic lighting"


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
  parser.add_argument("--benchmark", action="store_true", help="Time the encode pass.")
  return parser.parse_args()


def build_encoder(device):
  return T5EncoderModel(
    text_len=TEXT_LEN,
    dtype=DTYPE,
    device=device,
    checkpoint_path=CHECKPOINT_PATH,
    tokenizer_path=TOKENIZER_PATH,
  )


def encode_once(encoder, prompt, device):
  with torch.inference_mode():
    return encoder([prompt], device)[0]


def benchmark(encoder, prompt, device, run_count=10):
  for _ in range(3):
    _ = encode_once(encoder, prompt, device)
  torch.cuda.synchronize()

  times = []
  for _ in range(run_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = encode_once(encoder, prompt, device)
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

  encoder = build_encoder(device)

  if args.benchmark:
    benchmark(encoder, args.prompt, device)
    return

  context = encode_once(encoder, args.prompt, device)
  print(f"Prompt: {args.prompt!r}")
  print(f"Context shape: {tuple(context.shape)}, dtype: {context.dtype}")


if __name__ == "__main__":
  main()
