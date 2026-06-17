import torch
from sgl_kernel import fp8_scaled_mm

from wan.layers.quantization.fp8_kernel import sglang_per_token_quant_fp8, static_quant_fp8, triton_scaled_mm


def _process_scaled_mm_output(output, input_2d_shape, output_shape):
  if type(output) is tuple and len(output) == 2:
    output = output[0]
  return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)

# Input scaling factors are no longer optional in _scaled_mm starting
# from pytorch 2.5. Allocating a dummy tensor to pass as input_scale
TORCH_DEVICE_IDENTITY = None


def _apply_fallback_scaled_mm(
  qinput,
  weight,
  x_scale,
  weight_scale,
  input_2d_shape,
  output_shape,
  bias,
  input_dtype,
):
  global TORCH_DEVICE_IDENTITY
  if TORCH_DEVICE_IDENTITY is None:
    TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32, device=weight.device)

  output = torch._scaled_mm(
    qinput,
    weight,
    scale_a=TORCH_DEVICE_IDENTITY,
    scale_b=TORCH_DEVICE_IDENTITY,
    out_dtype=torch.float32,
  )

  output = _process_scaled_mm_output(output, input_2d_shape, output_shape)
  x_scale = torch.narrow(x_scale, 0, 0, input_2d_shape[0])

  output = output * x_scale * weight_scale.t()
  if bias is not None:
    output = output + bias
  return output.to(dtype=input_dtype)


def apply_fp8_linear(
  input: torch.Tensor,
  weight: torch.Tensor,
  weight_scale: torch.Tensor,
  input_scale: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
) -> torch.Tensor:
  input_2d = input.view(-1, input.shape[-1])
  output_shape = [*input.shape[:-1], weight.shape[1]]

  if input_scale is not None:
    assert input_scale.numel() == 1

    qinput, x_scale = static_quant_fp8(input_2d, input_scale, repeat_scale=True)
  else:
    qinput, x_scale = sglang_per_token_quant_fp8(input_2d)

  if weight_scale.numel() == weight.shape[1]:
    cutlass_compatible_b = weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
    if not cutlass_compatible_b:
      # Massage the input to be 2D
      qinput = qinput.view(-1, qinput.shape[-1])
      output = triton_scaled_mm(qinput, weight, x_scale, weight_scale, input.dtype, bias)
    else:
      output = fp8_scaled_mm(
        qinput,
        weight,
        x_scale,
        weight_scale,
        out_dtype=input.dtype,
        bias=bias,
      )
    return output.view(*output_shape)

  # torch.scaled_mm supports per tensor weights + activations only
  # so fallback to naive if per channel or per token
  per_tensor_weights = weight_scale.numel() == 1
  # When the number of token is 1,
  # per-token scale has shape (1, 1), per-tensor scale has shape (1) or ().
  per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2

  if per_tensor_weights and per_tensor_activations:
    # Fused GEMM_DQ; _scaled_mm with torch.compile requires len(weight_scale.shape) == len(x_scale.shape)
    if weight_scale.ndim == 0 and x_scale.ndim == 1:
      weight_scale = weight_scale.unsqueeze(0)
    output = torch._scaled_mm(
      qinput,
      weight,
      out_dtype=input.dtype,
      scale_a=x_scale,
      scale_b=weight_scale,
      bias=bias,
    )
    return _process_scaled_mm_output(output, input_2d.shape, output_shape)

  # Fallback for channelwise case, where we use unfused DQ
  # due to limitations with scaled_mm

  # Symmetric quantized GEMM by definition computes the following:
  #   C = (s_x * X) (s_w * W) + bias
  # This is equivalent to dequantizing the weights and activations
  # before applying a GEMM.
  #
  # In order to compute quantized operands, a quantized kernel
  # will rewrite the above like so:
  #   C = s_w * s_x * (X * W) + bias
  #
  # For the scaled_mm fallback case, we break this down, since it
  # does not support s_w being a vector.
  return _apply_fallback_scaled_mm(
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d.shape,
    output_shape,
    bias,
    input.dtype,
  )
