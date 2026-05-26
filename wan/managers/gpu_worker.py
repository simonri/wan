import gc

import torch

from wan.managers.scheduler import Scheduler
from wan.server_args import ServerArgs


def _oom_exceptions():
  types = [torch.cuda.OutOfMemoryError]
  if hasattr(torch, "OutOfMemoryError"):
    types.append(torch.OutOfMemoryError)
  return tuple(types)


def run_scheduler_process(server_args: ServerArgs) -> None:
  try:
    scheduler = Scheduler(server_args)
    scheduler.event_loop()
  except _oom_exceptions() as e:
    print(f"GPU OOM: {e}")
    raise
  finally:
    gc.collect()
    if torch.cuda.is_initialized():
      torch.cuda.empty_cache()
    print("Worker: Shutdown complete")
