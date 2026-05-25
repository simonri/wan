import os
import time

import torch


class DiffusionProfiler:
  """
  A wrapper around torch.profiler
  """

  _instance = None

  def __init__(self, full_profile: bool = False, num_steps: int | None = None, num_inference_steps: int | None = None):
    self.full_profile = full_profile
    self.log_dir = "./logs"

    os.makedirs(self.log_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
      activities.append(torch.profiler.ProfilerActivity.CUDA)

    common_torch_profiler_args = dict(activities=activities, record_shapes=True, with_stack=True, on_trace_ready=None)

    if self.full_profile:
      self.profiler = torch.profiler.profile(**common_torch_profiler_args)
      self.profile_mode_id = "full stages"
    else:
      warmup = 1
      num_actual_steps = num_inference_steps if num_steps == -1 else num_steps
      self.num_active_steps = num_actual_steps - warmup
      self.profiler = torch.profiler.profile(
        **common_torch_profiler_args,
        schedule=torch.profiler.schedule(skip_first=0, wait=0, warmup=warmup, active=self.num_active_steps, repeat=1),
      )
      self.profile_mode_id = f"{num_actual_steps} steps"

    self.has_stopped = False

    DiffusionProfiler._instance = self
    self.start()

  @classmethod
  def get_instance(cls) -> "DiffusionProfiler":
    return cls._instance

  def start(self):
    print("Starting profiler...")
    self.profiler.start()

  def stop(self, export_trace: bool = True):
    if self.has_stopped:
      return

    self.has_stopped = True
    print("Stopping profiler...")
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    self.profiler.stop()

    if export_trace:
      self._export_trace()

    DiffusionProfiler._instance = None

  def _step(self):
    self.profiler.step()

  def step_denoising_step(self):
    if not self.full_profile:
      if self.num_active_steps >= 0:
        self._step()
        self.num_active_steps -= 1
      else:
        self.stop()

  def _export_trace(self):
    try:
      os.makedirs("traces", exist_ok=True)
      sanitized_profile_mode_id = self.profile_mode_id.replace(" ", "_")

      time_unix_ms = int(time.time())
      trace_path = os.path.abspath(
        os.path.join(self.log_dir, f"trace_{sanitized_profile_mode_id}_{time_unix_ms}.trace.json.gz")
      )
      self.profiler.export_chrome_trace(trace_path)

      print(f"Exported trace to {trace_path}")
    except Exception as e:
      print(f"Error exporting trace: {e}")
