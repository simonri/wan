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

  # assume per tensor weights

  # Fused GEMM_DQ; _scaled_mm with torch.compile requires len(weight_scale.shape) == len(x_scale.shape)
  if weight_scale.ndim == 0 and x_scale.ndim == 1:
    weight_scale = weight_scale.unsqueeze(0)
  output = torch._scaled_mm(
    qinput,
    w,
    out_dtype=input.dtype,
    scale_a=x_scale,
    scale_b=weight_scale,
    bias=bias,
  )
  return _process_scaled_mm_output(output, input_2d.shape, output_shape)
