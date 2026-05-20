import argparse
import statistics

import torch

from wan.bench.nvtx_marker import NVTXMarker, cuda_profiler_start, cuda_profiler_stop
from wan.configs.models.encoders.t5 import T5Config
from wan.modules.t5 import T5EncoderModel
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.schedule_batch import Req
from wan.stages.text_encoding import TextEncodingStage

CHECKPOINT_PATH = "models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
DTYPE = torch.bfloat16
DEFAULT_PROMPT = "A cat playing piano in a cozy living room, cinematic lighting"


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
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


def encode_once(stage, prompt, pipeline_config, device):
  with torch.inference_mode():
    embeds_list, *_ = stage.encode_text(prompt, pipeline_config, device)
    return embeds_list[0]


def warmup(stage, prompt, pipeline_config, device, run_count=3):
  for _ in range(run_count):
    _ = encode_once(stage, prompt, pipeline_config, device)
  torch.cuda.synchronize()


def benchmark(stage, prompt, pipeline_config, device, run_count=10):
  warmup(stage, prompt, pipeline_config, device)

  times = []
  for _ in range(run_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = encode_once(stage, prompt, pipeline_config, device)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)

  print(f"Runs: {run_count}")
  print(f"Min:    {min(times):.6f} sec")
  print(f"Median: {statistics.median(times):.6f} sec")
  print(f"Mean:   {statistics.mean(times):.6f} sec")
  print(f"Std:    {statistics.pstdev(times):.6f} sec")


def profile_nsys(stage, prompt, pipeline_config, device):
  warmup(stage, prompt, pipeline_config, device, run_count=2)
  cuda_profiler_start()
  with NVTXMarker(stage.text_encoder.model):
    torch.cuda.nvtx.range_push("t5_encode")
    _ = encode_once(stage, prompt, pipeline_config, device)
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

  t5_config = T5Config()

  tokenizer = HuggingfaceTokenizer(name="google/umt5-xxl", seq_len=t5_config.arch_config.text_len)

  encoder = T5EncoderModel(
    config=t5_config,
    dtype=DTYPE,
    checkpoint_path=CHECKPOINT_PATH,
  )

  text_encoding_stage = TextEncodingStage(text_encoder=encoder, tokenizer=tokenizer)

  batch = Req(prompt=args.prompt)
  server_args = ServerArgs()

  text_encoding_stage(batch, server_args)

  if args.nsys:
    profile_nsys(text_encoding_stage, args.prompt, server_args.pipeline_config, local_torch_device)
    return

  if args.benchmark:
    benchmark(text_encoding_stage, args.prompt, server_args.pipeline_config, local_torch_device)
    return

  context = encode_once(text_encoding_stage, args.prompt, server_args.pipeline_config, local_torch_device)
  print(f"Prompt: {args.prompt!r}")
  print(f"Context shape: {tuple(context.shape)}, dtype: {context.dtype}")

  if args.compare:
    compare_context(context, args.compare)

  if args.save:
    save_context(context, args.save)


if __name__ == "__main__":
  main()
