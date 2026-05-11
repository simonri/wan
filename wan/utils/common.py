from collections.abc import Callable

from torch.library import Library

sglang_lib = Library("sglang", "FRAGMENT")


def direct_register_custom_op(
  op_name: str,
  op_func: Callable,
  mutates_args: list[str],
  fake_impl: Callable | None = None,
  target_lib: Library | None = None,
) -> None:
  """
  NOTE: Please try to use `register_custom_op` instead of this function.
  See `python/sglang/srt/utils/custom_op.py` for details.

  `torch.library.custom_op` can have significant overhead because it
  needs to consider complicated dispatching logic. This function
  directly registers a custom op and dispatches it to the CUDA backend.
  See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
  for more details.

  By default, the custom op is registered to the vLLM library. If you
  want to register it to a different library, you can pass the library
  object to the `target_lib` argument.

  IMPORTANT: the lifetime of the operator is tied to the lifetime of the
  library object. If you want to bind the operator to a different library,
  make sure the library object is alive when the operator is used.

  Note: This function will silently skip registration if the operator
  with the same name is already registered to avoid RuntimeError in
  multi-engine scenarios (e.g., VERL framework).
  """
  import torch.library

  my_lib = target_lib or sglang_lib

  # Check if operator is already registered to avoid duplicate registration
  # This is important for scenarios where multiple SGLang engines run in the same process
  try:
    # Try to access the operator to see if it's already registered
    lib_name = my_lib.m.name if hasattr(my_lib.m, "name") else "sglang"
    if hasattr(torch.ops, lib_name) and hasattr(getattr(torch.ops, lib_name), op_name):
      # Operator already exists, skip registration
      return
  except (AttributeError, RuntimeError):
    # Operator doesn't exist, proceed with registration
    pass

  if hasattr(torch.library, "infer_schema"):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
  else:
    # for pytorch 2.4
    import torch._custom_op.impl

    schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

  try:
    my_lib.define(op_name + schema_str)
    my_lib.impl(op_name, op_func, "CUDA")
    if fake_impl is not None:
      my_lib._register_fake(op_name, fake_impl)
  except RuntimeError as error:
    if "Tried to register an operator" in str(error) and "multiple times" in str(error):
      # Silently ignore duplicate registration errors
      # This can happen in multi-engine scenarios
      pass
    else:
      # Re-raise other RuntimeErrors
      raise error
  except AttributeError as error:
    # Always re-raise AttributeError as it indicates missing dependencies
    raise error
