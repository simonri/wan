import torch
import triton
import triton.language as tl


@triton.autotune(
  configs=[
    triton.Config({"BLOCK_N": 64}, num_warps=2),
    triton.Config({"BLOCK_N": 128}, num_warps=4),
    triton.Config({"BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_N": 512}, num_warps=4),
    triton.Config({"BLOCK_N": 1024}, num_warps=8),
  ],
  key=["inner_dim"],
)
@triton.jit
def _fused_scale_shift_4d_kernel(
  output_ptr,
  normalized_ptr,
  scale_ptr,
  shift_ptr,
  scale_constant: tl.constexpr,  # scale_constant is either 0 or 1.
  rows,
  inner_dim,
  seq_len,
  num_frames,
  frame_seqlen,
  BLOCK_N: tl.constexpr,
):
  pid_row = tl.program_id(0)
  pid_col = tl.program_id(1)

  col_offsets = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
  mask = col_offsets < inner_dim

  # Pointers for normalized and output
  row_base = pid_row * inner_dim
  norm_ptrs = normalized_ptr + row_base + col_offsets
  out_ptrs = output_ptr + row_base + col_offsets

  # Pointers for scale (per-frame) and shift (per-token)
  b_idx = pid_row // seq_len
  t_idx = pid_row % seq_len
  frame_idx_in_batch = t_idx // frame_seqlen

  scale_row_idx = b_idx * num_frames + frame_idx_in_batch
  scale_ptrs = scale_ptr + scale_row_idx * inner_dim + col_offsets
  # shift is per-token [B*L, C], indexed by pid_row directly
  shift_ptrs = shift_ptr + pid_row * inner_dim + col_offsets

  normalized = tl.load(norm_ptrs, mask=mask, other=0.0)
  scale = tl.load(scale_ptrs, mask=mask, other=0.0)
  shift = tl.load(shift_ptrs, mask=mask, other=0.0)

  scale_const_tensor = tl.full([BLOCK_N], scale_constant, dtype=scale.dtype)
  output = normalized * (scale_const_tensor + scale) + shift

  tl.store(out_ptrs, output, mask=mask)


@triton.jit
def fuse_scale_shift_kernel_blc_opt(
  x_ptr,
  shift_ptr,
  scale_ptr,
  scale_constant: tl.constexpr,  # scale_constant is either 0 or 1.,
  y_ptr,
  B,
  L,
  C,
  stride_x_b,
  stride_x_l,
  stride_x_c,
  stride_s_b,
  stride_s_l,
  stride_s_c,
  stride_sc_b,
  stride_sc_l,
  stride_sc_c,
  SCALE_IS_SCALAR: tl.constexpr,
  SHIFT_IS_SCALAR: tl.constexpr,
  BLOCK_L: tl.constexpr,
  BLOCK_C: tl.constexpr,
):
  pid_l = tl.program_id(0)
  pid_c = tl.program_id(1)
  pid_b = tl.program_id(2)

  l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
  c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

  mask_l = l_offsets < L
  mask_c = c_offsets < C
  mask = mask_l[:, None] & mask_c[None, :]

  x_off = pid_b * stride_x_b + l_offsets[:, None] * stride_x_l + c_offsets[None, :] * stride_x_c
  x = tl.load(x_ptr + x_off, mask=mask, other=0)

  if SHIFT_IS_SCALAR:
    shift_val = tl.load(shift_ptr)
    shift = tl.full((BLOCK_L, BLOCK_C), shift_val, dtype=shift_val.dtype)
  else:
    s_off = pid_b * stride_s_b + l_offsets[:, None] * stride_s_l + c_offsets[None, :] * stride_s_c
    shift = tl.load(shift_ptr + s_off, mask=mask, other=0)

  if SCALE_IS_SCALAR:
    scale_val = tl.load(scale_ptr)
    scale = tl.full((BLOCK_L, BLOCK_C), scale_val, dtype=scale_val.dtype)
  else:
    sc_off = pid_b * stride_sc_b + l_offsets[:, None] * stride_sc_l + c_offsets[None, :] * stride_sc_c
    scale = tl.load(scale_ptr + sc_off, mask=mask, other=0)

  y = x * (scale_constant + scale) + shift
  tl.store(y_ptr + x_off, y, mask=mask)


def fuse_scale_shift_kernel(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  scale_constant: float = 1.0,
  block_l: int = 128,
  block_c: int = 128,
):
  assert (x.is_cuda and scale.is_cuda) or (x.is_xpu and scale.is_xpu)
  assert x.is_contiguous()

  B, L, C = x.shape
  output = torch.empty_like(x)

  if scale.dim() == 4:
    # scale/shift: [B, F, 1, C]
    rows = B * L
    x_2d = x.view(rows, C)
    output_2d = output.view(rows, C)

    def grid(meta):
      return (rows, triton.cdiv(C, meta["BLOCK_N"]))

    num_frames = scale.shape[1]
    assert L % num_frames == 0, "seq_len must be divisible by num_frames for 4D scale/shift"
    frame_seqlen = L // num_frames

    # Compact scale [B, F, 1, C] -> [B*F, C] (per-frame)
    scale_reshaped = scale.squeeze(2).reshape(-1, C).contiguous()
    # shift is per-token [B, L, C] -> [B*L, C]
    shift_reshaped = shift.reshape(rows, C).contiguous()

    _fused_scale_shift_4d_kernel[grid](
      output_2d,
      x_2d,
      scale_reshaped,
      shift_reshaped,
      scale_constant,
      rows,
      C,
      L,
      num_frames,
      frame_seqlen,
    )
  else:
    # 2D: [B, C] or [1, C]  -> treat as [B, 1, C] and broadcast over L
    # 3D: [B, L, C] (or broadcastable variants like [B, 1, C], [1, L, C], [1, 1, C])
    # Also support scalar (0D or 1-element)
    if scale.dim() == 0 or (scale.dim() == 1 and scale.numel() == 1):
      scale_blc = scale.reshape(1)
    elif scale.dim() == 2:
      scale_blc = scale[:, None, :]
    elif scale.dim() == 3:
      scale_blc = scale
    else:
      raise ValueError("scale must be 0D/1D(1)/2D/3D or 4D")

    if shift.dim() == 0 or (shift.dim() == 1 and shift.numel() == 1):
      shift_blc = shift.reshape(1)
    elif shift.dim() == 2:
      shift_blc = shift[:, None, :]
    elif shift.dim() == 3:
      shift_blc = shift
    else:
      # broadcast later via expand if possible
      shift_blc = shift

    need_scale_scalar = scale_blc.dim() == 1 and scale_blc.numel() == 1
    need_shift_scalar = shift_blc.dim() == 1 and shift_blc.numel() == 1

    if not need_scale_scalar:
      scale_exp = scale_blc.expand(B, L, C)
      s_sb, s_sl, s_sc = scale_exp.stride()
    else:
      s_sb = s_sl = s_sc = 0

    if not need_shift_scalar:
      shift_exp = shift_blc.expand(B, L, C)
      sh_sb, sh_sl, sh_sc = shift_exp.stride()
    else:
      sh_sb = sh_sl = sh_sc = 0

    # If both scalars and both zero, copy fast-path
    if need_scale_scalar and need_shift_scalar:
      if not (scale_blc.any().to("cpu", non_blocking=True) or shift_blc.any().to("cpu", non_blocking=True)):
        output.copy_(x)
        return output

    grid = (triton.cdiv(L, block_l), triton.cdiv(C, block_c), B)
    fuse_scale_shift_kernel_blc_opt[grid](
      x,
      shift_blc if need_shift_scalar else shift_exp,
      scale_blc if need_scale_scalar else scale_exp,
      scale_constant,
      output,
      B,
      L,
      C,
      x.stride(0),
      x.stride(1),
      x.stride(2),
      sh_sb,
      sh_sl,
      sh_sc,
      s_sb,
      s_sl,
      s_sc,
      SCALE_IS_SCALAR=need_scale_scalar,
      SHIFT_IS_SCALAR=need_shift_scalar,
      BLOCK_L=block_l,
      BLOCK_C=block_c,
      num_warps=4,
      num_stages=2,
    )
  return output
