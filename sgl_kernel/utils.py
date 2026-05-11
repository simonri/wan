import functools

import cutlass
import torch

WARP_SIZE = 32

TORCH_TO_CUTE_DTYPE = {
  torch.float16: cutlass.Float16,
  torch.bfloat16: cutlass.BFloat16,
  torch.float32: cutlass.Float32,
}


def cache_once(fn):
  """
  NOTE: `functools.lru_cache` is not compatible with `torch.compile`
  So we manually implement a simple cache_once decorator to replace it.
  """
  result_map = {}

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    key = (args, tuple(sorted(kwargs.items())))
    if key not in result_map:
      result_map[key] = fn(*args, **kwargs)
    return result_map[key]

  return wrapper


@cache_once
def is_arch_support_pdl() -> bool:
  if bool(torch.version.hip):
    return False
  try:
    device = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device)
  except Exception:
    return False
  return major >= 9
