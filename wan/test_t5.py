import argparse
import statistics

import torch
from transformers import AutoTokenizer

from wan.bench.nvtx_marker import NVTXMarker, cuda_profiler_start, cuda_profiler_stop
from wan.configs.pipeline.wan import WanI2VConfig
from wan.modules.t5 import T5Encoder
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.schedule_batch import Req
from wan.stages.text_encoding import TextEncodingStage

CHECKPOINT_PATH = "models/text_encoders/umt5-xxl-enc-bf16.safetensors"
DTYPE = torch.bfloat16
DEFAULT_PROMPT = "A cat playing piano in a cozy living room, cinematic lighting"


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--benchmark", action="store_true", help="Time the encode pass.")
  parser.add_argument(
    "--nsys",
    action="store_true",
    help="Run one encode inside cudaProfilerStart/Stop with NVTX module ranges. Launch under `nsys profile --capture-range=cudaProfilerApi`.",
  )
  parser.add_argument("--save", type=str, default=None, help="Save the context to a file.")
  parser.add_argument("--compare", type=str, default=None, help="Compare the context to a file.")
  return parser.parse_args()


def save_context(context, save_path):
  torch.save(context, save_path)


def encode_once(stage, prompt, server_args, device):
  with torch.inference_mode():
    embeds = stage.encode_text(prompt, server_args, device)
    return embeds[0]


def warmup(stage, prompt, server_args, device, run_count=3):
  for _ in range(run_count):
    _ = encode_once(stage, prompt, server_args, device)
  torch.cuda.synchronize()


def benchmark(stage, prompt, server_args, device, run_count=10):
  warmup(stage, prompt, server_args, device)

  times = []
  for _ in range(run_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = encode_once(stage, prompt, server_args, device)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)

  print(f"Runs: {run_count}")
  print(f"Min:    {min(times):.6f} sec")
  print(f"Median: {statistics.median(times):.6f} sec")
  print(f"Mean:   {statistics.mean(times):.6f} sec")
  print(f"Std:    {statistics.pstdev(times):.6f} sec")


def profile_nsys(stage, prompt, server_args, device):
  warmup(stage, prompt, server_args, device, run_count=2)
  cuda_profiler_start()
  with NVTXMarker(stage.text_encoder):
    torch.cuda.nvtx.range_push("t5_encode")
    _ = encode_once(stage, prompt, server_args, device)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
  cuda_profiler_stop()


def compare_context(context, compare_path):
  compare_context = torch.load(compare_path)
  if torch.allclose(context, compare_context):
    print("Contexts are the same")
  else:
    print("Contexts are different")
    max_diff = (context - compare_context).abs().max().item()
    print(f"Max diff: {max_diff:.8f}")


def main():
  args = parse_args()
  local_torch_device = get_local_torch_device()

  pipeline_config = WanI2VConfig()
  server_args = ServerArgs(pipeline_config=pipeline_config)

  tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

  text_encoder = T5Encoder(config=pipeline_config.text_encoder_config)
  text_encoder.load(CHECKPOINT_PATH, server_args)

  text_encoding_stage = TextEncodingStage(text_encoder=text_encoder, tokenizer=tokenizer)

  batch = Req(prompt=DEFAULT_PROMPT)

  text_encoding_stage(batch, server_args)

  if args.nsys:
    profile_nsys(text_encoding_stage, DEFAULT_PROMPT, server_args, local_torch_device)
    return

  if args.benchmark:
    benchmark(text_encoding_stage, DEFAULT_PROMPT, server_args, local_torch_device)
    return

  context = encode_once(text_encoding_stage, DEFAULT_PROMPT, server_args, local_torch_device)
  print(f"Context shape: {tuple(context.shape)}, dtype: {context.dtype}")

  if args.compare:
    compare_context(context, args.compare)

  if args.save:
    save_context(context, args.save)


if __name__ == "__main__":
  main()
