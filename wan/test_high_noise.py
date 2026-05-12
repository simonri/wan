import argparse
import statistics

import torch

from wan.configs.pipeline.wan import WanI2VConfig

from .bench.layer_timer import LayerTimer
from .bench.nvtx_marker import NVTXMarker, cuda_profiler_start, cuda_profiler_stop
from .image2video import _HIGH_NOISE_I2V_CHECKPOINT, _load_i2v_wan_model, _merge_lora_into_wan_model

_HIGH_NOISE_LIGHTNING_LORA = "models/loras/lightning_high_noise_model.safetensors"

IMAGE_SIZE = (480, 832)
FRAME_NUM = 81
TIMESTEP = 950.0  # high-noise side of boundary (boundary*1000 = 875)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--benchmark", action="store_true", help="Time one denoising step.")
  parser.add_argument("--profile-layers", action="store_true", help="Time submodules during one step.")
  parser.add_argument(
    "--profile-parents", action="store_true", help="Include non-leaf parent modules in layer timings."
  )
  parser.add_argument("--profile-limit", type=int, default=40, help="Maximum number of timed layers to print.")
  parser.add_argument(
    "--profile-filter", type=str, default=None, help="Only profile modules whose name or class contains this substring."
  )
  parser.add_argument(
    "--nsys",
    action="store_true",
    help="Run one step inside cudaProfilerStart/Stop with NVTX module ranges. "
    + "Launch under `nsys profile --capture-range=cudaProfilerApi`.",
  )
  return parser.parse_args()


def build_model(device, dtype, config):
  model = _load_i2v_wan_model(_HIGH_NOISE_I2V_CHECKPOINT, config.arch_config)
  _merge_lora_into_wan_model(model, _HIGH_NOISE_LIGHTNING_LORA)
  model.eval().requires_grad_(False)
  model.to(dtype)
  model.to(device)
  return model


def build_inputs(device, config, dtype, seed=0):
  vae_stride = config.vae_stride  # (4, 8, 8)
  H, W = IMAGE_SIZE
  lat_h = H // vae_stride[1]
  lat_w = W // vae_stride[2]
  lat_f = (FRAME_NUM - 1) // vae_stride[0] + 1

  g = torch.Generator(device=device).manual_seed(seed)

  # noise latent: 16 channels
  noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=g, device=device)

  # y conditioning: 4 mask channels (first frame marked) + 16 reference latent channels
  msk = torch.zeros(4, lat_f, lat_h, lat_w, dtype=torch.float32, device=device)
  msk[:, 0] = 1.0
  y_latent = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=g, device=device)
  y = torch.cat([noise, msk, y_latent], dim=0)

  # context: T5 embeddings — bf16 to match real T5 output, ~16 token prompt, padded to text_len=512
  context = torch.randn(16, 4096, dtype=torch.bfloat16, generator=g, device=device)
  context = torch.cat([context, context.new_zeros(512 - context.size(0), context.size(1))])

  t = torch.tensor([TIMESTEP], dtype=torch.float32, device=device)

  return {
    "hidden_states": y.unsqueeze(0).to(dtype),
    "encoder_hidden_states": context.unsqueeze(0).to(dtype),
    "timestep": t,
  }


def step_once(model, inputs, dtype):
  with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
    return model(**inputs)


def warmup(model, inputs, dtype, run_count=2):
  for _ in range(run_count):
    _ = step_once(model, inputs, dtype)
  torch.cuda.synchronize()


def benchmark(model, inputs, dtype, run_count=10):
  times = []
  for _ in range(run_count):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = step_once(model, inputs, dtype)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) / 1000)

  print(f"Runs: {run_count}")
  print(f"Min:    {min(times):.6f} sec")
  print(f"Median: {statistics.median(times):.6f} sec")
  print(f"Mean:   {statistics.mean(times):.6f} sec")
  print(f"Std:    {statistics.pstdev(times):.6f} sec")


def profile_nsys(model, inputs, dtype):
  warmup(model, inputs, dtype, run_count=2)
  cuda_profiler_start()
  with NVTXMarker(model):
    torch.cuda.nvtx.range_push("step_once")
    _ = step_once(model, inputs, dtype)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
  cuda_profiler_stop()


def profile_layers(model, inputs, dtype, limit, name_filter, include_parents):
  warmup(model, inputs, dtype, run_count=1)

  with LayerTimer(model, name_filter=name_filter, include_parents=include_parents) as timer:
    _ = step_once(model, inputs, dtype)

  if include_parents:
    print("Note: parent module timings include child module time.")
  print(f"{'Total ms':>10}  {'Calls':>5}  {'Module':<22}  Name")
  print("-" * 90)
  for total_ms, calls, class_name, name in timer.results_ms()[:limit]:
    print(f"{total_ms:10.3f}  {calls:5d}  {class_name:<22}  {name}")


def main():
  args = parse_args()
  device = torch.device("cuda:0")
  dit_cfg = WanI2VConfig().dit_config
  dtype = dit_cfg.arch_config.param_dtype
  torch.backends.cudnn.benchmark = True

  model = build_model(device, dtype, dit_cfg)
  inputs = build_inputs(device, dit_cfg, dtype)

  if args.nsys:
    profile_nsys(model, inputs, dtype)
  elif args.profile_layers:
    profile_layers(model, inputs, dtype, args.profile_limit, args.profile_filter, args.profile_parents)
  elif args.benchmark:
    warmup(model, inputs, dtype)
    benchmark(model, inputs, dtype)
  else:
    out = step_once(model, inputs, dtype)
    print(f"Output shape: {tuple(out.shape)}, dtype: {out.dtype}")


if __name__ == "__main__":
  main()
