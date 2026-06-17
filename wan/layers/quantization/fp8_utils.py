import torch

from wan.layers.quantization.fp8_kernel import sglang_per_token_quant_fp8


def _process_scaled_mm_output(output, input_2d_shape, output_shape):
  if type(output) is tuple and len(output) == 2:
    output = output[0]
  return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)


def apply_fp8_linear(
  input: torch.Tensor,
  weight: torch.Tensor,
  weight_scale: torch.Tensor,
  bias: torch.Tensor | None = None,
) -> torch.Tensor:
  input_2d = input.view(-1, input.shape[-1])
  # weight is [out, in] (nn.Linear convention)
  # w = weight.T is [in, out] column-major — required by cuBLAS _scaled_mm
  w = weight.T
  output_shape = [*input.shape[:-1], weight.shape[0]]

  qinput, x_scale = sglang_per_token_quant_fp8(input_2d)

  # x_scale is per-token: shape (M, 1). RowWise scaling requires scale_b to be
  # (1, out_features) — broadcast the per-tensor scalar to that shape so the
  # fused kernel handles dequantization without a separate pass.
  ws = weight_scale.reshape(1, 1).expand(1, weight.shape[0]).contiguous()
  output = torch._scaled_mm(
    qinput,
    w,
    out_dtype=input.dtype,
    scale_a=x_scale,
    scale_b=ws,
    bias=bias,
  )
  return _process_scaled_mm_output(output, input_2d.shape, output_shape)
