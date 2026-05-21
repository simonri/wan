import contextlib

import torch
import torch.nn as nn

PRECISION_TO_TYPE = {
  "fp32": torch.float32,
  "fp16": torch.float16,
  "bf16": torch.bfloat16,
}


class skip_init_modules:
  def __enter__(self):
    # save originals
    self._orig_reset = {}
    for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
      self._orig_reset[cls] = cls.reset_parameters
      cls.reset_parameters = lambda self: None

  def __exit__(self, exc_type, exc_value, traceback):
    # restore originals
    for cls, orig in self._orig_reset.items():
      cls.reset_parameters = orig


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
  old_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(old_dtype)
