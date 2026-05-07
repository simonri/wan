import torch
import torch.nn as nn
import triton
import triton.language as tl


def _rmsnorm_configs():
  return [triton.Config({}, num_warps=w, num_stages=s) for w in (1, 2, 4, 8) for s in (2, 3)]


@triton.autotune(configs=_rmsnorm_configs(), key=["N"])
@triton.jit
def _rmsnorm_fwd_kernel(
  X,
  W,
  Y,
  Rstd,
  stride_x_row,
  N,
  eps,
  BLOCK_SIZE: tl.constexpr,
  STORE_RSTD: tl.constexpr,
  APPLY_SILU: tl.constexpr,
):
  row = tl.program_id(0)
  X += row * stride_x_row
  Y += row * stride_x_row

  cols = tl.arange(0, BLOCK_SIZE)
  mask = cols < N

  x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / N
  rstd = 1.0 / tl.sqrt(var + eps)
  if STORE_RSTD:
    tl.store(Rstd + row, rstd)

  w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
  y = x * rstd * w
  if APPLY_SILU:
    y = y * tl.sigmoid(y)
  tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
  DY,
  X,
  W,
  Rstd,
  DX,
  DW_partial,
  stride_x_row,
  M,
  N,
  BLOCK_SIZE: tl.constexpr,
):
  row = tl.program_id(0)
  X += row * stride_x_row
  DY += row * stride_x_row
  DX += row * stride_x_row

  cols = tl.arange(0, BLOCK_SIZE)
  mask = cols < N

  x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
  w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)

  # dx = rstd * (w * dy - x * mean(w * dy * x) * rstd^2)
  wdy = w * dy
  c = tl.sum(wdy * x, axis=0) / N
  dx = rstd * (wdy - x * c * rstd * rstd)
  tl.store(DX + cols, dx.to(DX.dtype.element_ty), mask=mask)

  # accumulate dw per row; final reduction in PyTorch
  dw_row = dy * x * rstd
  tl.store(DW_partial + row * N + cols, dw_row, mask=mask)


def _launch_fwd(x, weight, eps, store_rstd, apply_silu):
  x_shape = x.shape
  x = x.contiguous()
  x_flat = x.view(-1, x_shape[-1])
  M, N = x_flat.shape

  y = torch.empty_like(x_flat)
  # 1-element dummy when STORE_RSTD=False; the constexpr branch elides the store.
  rstd = torch.empty(M if store_rstd else 1, device=x.device, dtype=torch.float32)

  BLOCK_SIZE = triton.next_power_of_2(N)

  _rmsnorm_fwd_kernel[(M,)](
    x_flat,
    weight,
    y,
    rstd,
    x_flat.stride(0),
    N,
    eps,
    BLOCK_SIZE=BLOCK_SIZE,
    STORE_RSTD=store_rstd,
    APPLY_SILU=apply_silu,
  )

  return y.view(x_shape), rstd, x_flat


class _RMSNormFn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, weight, eps):
    y, rstd, x_flat = _launch_fwd(x, weight, eps, store_rstd=True, apply_silu=False)
    ctx.save_for_backward(x_flat, weight, rstd)
    ctx.x_shape = x.shape
    return y

  @staticmethod
  def backward(ctx, dy):
    x_flat, weight, rstd = ctx.saved_tensors
    M, N = x_flat.shape
    dy = dy.contiguous().view(-1, N)

    dx = torch.empty_like(x_flat)
    dw_partial = torch.empty((M, N), device=x_flat.device, dtype=torch.float32)
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 16)

    _rmsnorm_bwd_kernel[(M,)](
      dy,
      x_flat,
      weight,
      rstd,
      dx,
      dw_partial,
      x_flat.stride(0),
      M,
      N,
      BLOCK_SIZE=BLOCK_SIZE,
      num_warps=num_warps,
    )

    dw = dw_partial.sum(dim=0).to(weight.dtype)
    return dx.view(ctx.x_shape), dw, None


def rms_norm_triton(x, weight, eps=1e-6, silu=False):
  needs_autograd = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
  if needs_autograd:
    y = _RMSNormFn.apply(x, weight, eps)
    if silu:
      y = y * torch.sigmoid(y)
    return y
  y, _, _ = _launch_fwd(x, weight, eps, store_rstd=False, apply_silu=silu)
  return y


class TritonRMSNorm(nn.Module):
  """Drop-in replacement for the last-dim RMSNorm path."""

  def __init__(self, dim, eps=1e-6, bias=False):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))
    self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

  def forward(self, x):
    y = rms_norm_triton(x, self.gamma, self.eps)
    return y + self.bias if self.bias is not None else y
