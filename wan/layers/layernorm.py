import torch
from torch import nn
from torch.nn import functional as F

from wan.jit_kernel.elementwise import fused_add_rmsnorm, rmsnorm
from wan.kernels.rmsnorm_onepass import triton_one_pass_rms_norm
from wan.kernels.scale_shift import fuse_scale_shift_kernel_blc_opt as fuse_scale_shift_kernel
from wan.layers.custom_op import CustomOp


# Copied and adapted from sglang
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
  """Root mean square normalization.

  Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
  Refer to https://arxiv.org/abs/1910.07467
  """

  def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    dtype: torch.dtype = torch.float32,
    var_hidden_size: int | None = None,
  ) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.hidden_size = hidden_size
    self.variance_size_override = None if var_hidden_size == hidden_size else var_hidden_size

  def forward_cuda(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if residual is not None:
      residual_shape = residual.shape
      residual = residual.view(-1, shape[-1])

    if x.dtype == torch.float:
      if residual is None and self.variance_size_override is None:
        return self.forward_native(x).view(shape)
      out = self.forward_triton(x, residual)
      if residual is not None:
        return out[0].view(shape), out[1].view(residual_shape)
      out = out.view(shape)
      return out
    elif self.variance_size_override is not None:
      return self.forward_native(x, residual)
    elif residual is not None:
      fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
      return x.view(shape), residual.view(residual_shape)
    else:
      if x.shape[-1] <= 128:
        out = triton_one_pass_rms_norm(x, self.weight.data, self.variance_epsilon)
      else:
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    out = out.view(shape)

    return out

  def _get_weight(self, dtype: torch.dtype) -> torch.Tensor:
    """Return weight matched to *dtype*.

    MUSA kernels require input and weight to share the same dtype,
    unlike CUDA kernels which may handle mixed dtypes internally.
    """
    weight = self.weight.data
    if weight.dtype != dtype:
      weight = weight.to(dtype=dtype)
    return weight

  def extra_repr(self) -> str:
    return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"


class FP32LayerNorm(nn.LayerNorm):
  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    origin_dtype = inputs.dtype
    device = inputs.device
    return F.layer_norm(
      inputs.float(),
      self.normalized_shape,
      self.weight.float().to(device=device) if self.weight is not None else None,
      self.bias.float().to(device=device) if self.bias is not None else None,
      self.eps,
    ).to(origin_dtype)


def _ensure_contiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
  return tensor.contiguous() if tensor is not None else None


class _ScaleResidualNormScaleShift(CustomOp):
  """
  Fused kernel that combines:
  1. residual_out = residual + gate * x
  2. normed = layernorm(residual_out) or rmsnorm(residual_out)
  3. out = normed * (1 + scale) + shift
  compute_dtype is always fp32 for higher precision.
  """

  norm_type: str

  def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    elementwise_affine: bool = False,
    dtype: torch.dtype = torch.float32,
    prefix: str = "",
  ):
    super().__init__()
    self.eps = eps
    self.dtype = dtype
    if self.norm_type == "rms":
      self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
    elif self.norm_type == "layer":
      self.norm = FP32LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)
    else:
      raise NotImplementedError(f"Norm type {self.norm_type} not implemented")

  def forward_cuda(
    self,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % 256 != 0 and x.shape[-1] <= 8192:
      import warnings

      warnings.warn(
        "FusedScaleResidualNormScaleShift cuda not available, using native fallback",
        stacklevel=2,
      )
      return self.forward_native(residual, x, gate, shift, scale)

    from wan.jit_kernel.scale_residual_norm_scale_shift import fused_scale_residual_norm_scale_shift

    if isinstance(gate, int) and gate != 1:
      raise ValueError(f"Only gate value of 1 is supported for int type, but got {gate}")

    return fused_scale_residual_norm_scale_shift(
      residual.contiguous(),
      x.contiguous(),
      gate.contiguous() if isinstance(gate, torch.Tensor) else None,
      _ensure_contiguous(getattr(self.norm, "weight", None)),
      _ensure_contiguous(getattr(self.norm, "bias", None)),
      scale.contiguous(),
      shift.contiguous(),
      self.norm_type,
      self.eps,
    )

  def forward_hip(self, *args, **kwargs):
    # ROCm does not support CUDA/CUTLASS-based fused kernels yet,
    # so we fall back to the native PyTorch implementation.
    return self.forward_native(*args, **kwargs)

  def forward_musa(self, *args, **kwargs):
    # MUSA does not support CUDA/CUTLASS-based fused kernels yet,
    # so we fall back to the native PyTorch implementation.
    return self.forward_native(*args, **kwargs)

  def forward_xpu(self, *args, **kwargs):
    # XPU does not support CUDA/CUTLASS-based fused kernels yet,
    # so we fall back to the native PyTorch implementation.
    return self.forward_native(*args, **kwargs)

  def forward_native(
    self,
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor | int,
    shift: torch.Tensor,
    scale: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # x.shape: [batch_size, seq_len, inner_dim]
    if isinstance(gate, int):
      # used by cross-attention, should be 1
      assert gate == 1
      residual_output = residual + x
    elif isinstance(gate, torch.Tensor):
      if gate.dim() == 4:
        # gate.shape: [batch_size, num_frames, 1, inner_dim]
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        residual_output = residual + (x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate).flatten(1, 2)
      else:
        # gate.shape: [batch_size, 1, inner_dim]
        residual_output = residual + x * gate
    else:
      raise ValueError(f"Gate type {type(gate)} not supported")
    normalized = self.norm(residual_output)
    modulated = fuse_scale_shift_kernel(normalized, scale, shift)
    return modulated, residual_output


class _NormScaleShift(CustomOp):
  """
  Fused kernel that combines:
  1. normed = layernorm(x) or rmsnorm(x)
  2. out = normed * (1 + scale) + shift
  compute_dtype is always fp32 for higher precision.
  """

  norm_type: str

  def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    elementwise_affine: bool = False,
    dtype: torch.dtype = torch.float32,
    prefix: str = "",
  ):
    super().__init__()
    self.eps = eps
    if self.norm_type == "rms":
      self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
    elif self.norm_type == "layer":
      self.norm = FP32LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype)
    else:
      raise NotImplementedError(f"Norm type {self.norm_type} not implemented")

  def forward_cuda(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 256 != 0 and x.shape[-1] <= 8192:
      import warnings

      warnings.warn(
        "FusedNormScaleShift cuda not available, using native fallback",
        stacklevel=2,
      )
      return self.forward_native(x, shift, scale)

    from wan.jit_kernel.scale_residual_norm_scale_shift import fused_norm_scale_shift

    return fused_norm_scale_shift(
      x.contiguous(),
      _ensure_contiguous(getattr(self.norm, "weight", None)),
      _ensure_contiguous(getattr(self.norm, "bias", None)),
      scale.contiguous(),
      shift.contiguous(),
      self.norm_type,
      self.eps,
    )

  def forward_native(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    normalized = self.norm(x)
    modulated = fuse_scale_shift_kernel(normalized, scale, shift)
    return modulated.to(x.dtype)


class LayerNormScaleShift(_NormScaleShift):
  norm_type = "layer"


class ScaleResidualLayerNormScaleShift(_ScaleResidualNormScaleShift):
  norm_type = "layer"
