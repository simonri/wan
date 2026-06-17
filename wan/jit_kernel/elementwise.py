import torch

from sgl_kernel.utils import is_arch_support_pdl

try:
  import flashinfer.norm as _flashinfer_norm

  _has_flashinfer = True
except ImportError:
  _has_flashinfer = False

_FLASHINFER_NORM_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _fused_add_rmsnorm_internal(
  input: torch.Tensor,
  residual: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  enable_pdl: bool | None,
) -> None:
  if enable_pdl is None:
    enable_pdl = is_arch_support_pdl()
  torch.ops.sgl_kernel.fused_add_rmsnorm.default(input, residual, weight, eps, enable_pdl)


def fused_add_rmsnorm(
  input: torch.Tensor,
  residual: torch.Tensor,
  weight: torch.Tensor,
  eps: float = 1e-6,
  enable_pdl: bool | None = None,
) -> None:
  r"""Fused add root mean square normalization.

  Step 1:
  ``residual[i] += input[i]``

  Step 2:
  ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

  Parameters
  ----------
  input: torch.Tensor
      Input tensor, shape (batch_size, hidden_size).
  residual: torch.Tensor
      Residual tensor, shape (batch_size, hidden_size).
  weight: torch.Tensor
      Weight tensor, shape (hidden_size,).
  eps: float
      Epsilon for numerical stability.
  enable_pdl: Optional[bool]
      Whether to enable `programmatic dependent launch
      <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
      If None, will be automatically enabled on Hopper architecture.
  """
  if _has_flashinfer and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES and not torch.compiler.is_dynamo_compiling():
    _flashinfer_norm.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
  else:
    _fused_add_rmsnorm_internal(input, residual, weight, eps, enable_pdl)


def _rmsnorm_internal(
  input: torch.Tensor,
  weight: torch.Tensor,
  eps: float,
  out: torch.Tensor | None,
  enable_pdl: bool | None,
) -> torch.Tensor:
  if out is None:
    out = torch.empty_like(input)
  if enable_pdl is None:
    enable_pdl = is_arch_support_pdl()
  torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
  return out


# These implementations extensively draw from and build upon the FlashInfer project https://github.com/flashinfer-ai/flashinfer
# Kudos to @yzh119
def rmsnorm(
  input: torch.Tensor,
  weight: torch.Tensor,
  eps: float = 1e-6,
  out: torch.Tensor | None = None,
  enable_pdl: bool | None = None,
) -> torch.Tensor:
  r"""Root mean square normalization.

  ``out[i] = (input[i] / RMS(input)) * weight[i]``

  Parameters
  ----------
  input: torch.Tensor
      Input tensor, shape (batch_size, hidden_size).
  weight: torch.Tensor
      Weight tensor, shape (hidden_size,).
  eps: float
      Epsilon for numerical stability.
  out: Optional[torch.Tensor]
      The output tensor, if specified, the kernel will update this tensor inplace.
  enable_pdl: Optional[bool]
      Whether to enable `programmatic dependent launch
      <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
      If None, will be automatically enabled on Hopper architecture.

  Returns
  -------
  output: torch.Tensor
      Normalized tensor, shape (batch_size, hidden_size).
  """
  # torch.compiler.is_dynamo_compiling(): FlashInfer norm paths are not safe under
  # torch.compile(..., fullgraph=True). Dynamo traces into FlashInfer's JIT module
  # loading path, which calls Path.exists() / os.stat() — both untraceable — causing
  # the entire compilation to fail. We fall back to the internal implementation while
  # tracing as a temporary workaround. Once the upstream fix is merged and we upgrade
  # FlashInfer, this check can be removed.
  # See: https://github.com/flashinfer-ai/flashinfer/issues/2734
  #      https://github.com/flashinfer-ai/flashinfer/pull/2733
  if _has_flashinfer and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES and not torch.compiler.is_dynamo_compiling():
    return _flashinfer_norm.rmsnorm(input, weight, eps, out, enable_pdl)
  else:
    return _rmsnorm_internal(input, weight, eps, out, enable_pdl)
