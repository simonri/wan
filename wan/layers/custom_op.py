from collections.abc import Callable
from typing import Any

from torch import nn


class CustomOp(nn.Module):
  """
  Base class for custom ops.
  Dispatches the forward method to the appropriate backend.
  """

  def __init__(self) -> None:
    super().__init__()
    self._forward_method = self.dispatch_forward()

  def forward(self, *args, **kwargs) -> Any:
    return self._forward_method(*args, **kwargs)

  def forward_native(self, *args, **kwargs) -> Any:
    """PyTorch-native implementation of the forward method.
    This method is optional. If implemented, it can be used with compilers
    such as torch.compile or PyTorch XLA. Also, it can be used for testing
    purposes.
    """
    raise NotImplementedError

  def forward_cuda(self, *args, **kwargs) -> Any:
    raise NotImplementedError

  def dispatch_forward(self) -> Callable:
    return self.forward_cuda

  @classmethod
  def enabled(cls) -> bool:
    # since we are not using Inductor, we always return True
    return True

  @staticmethod
  def default_on() -> bool:
    """
    On by default if level < CompilationLevel.PIECEWISE
    Specifying 'all' or 'none' in custom_op takes precedence.
    """
    raise NotImplementedError

  # Dictionary of all custom ops (classes, indexed by registered name).
  # To check if an op with a name is enabled, call .enabled() on the class.
  # Examples:
  # - MyOp.enabled()
  # - op_registry["my_op"].enabled()
  op_registry: dict[str, type["CustomOp"]] = {}

  # Decorator to register custom ops.
  @classmethod
  def register(cls, name: str) -> Callable:

    def decorator(op_cls):
      assert name not in cls.op_registry, f"Duplicate op name: {name}"
      op_cls.name = name
      cls.op_registry[name] = op_cls
      return op_cls

    return decorator
