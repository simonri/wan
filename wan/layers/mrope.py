import functools

import torch
import torch.nn as nn


def _to_tuple(x: int | tuple[int, ...], dim: int = 2) -> tuple[int, ...]:
  if isinstance(x, int):
    return (x,) * dim
  elif len(x) == dim:
    return x
  else:
    raise ValueError(f"Expected length {dim} or int, but got {x}")


class OneDRotaryEmbedding(torch.nn.Module):
  """1D rotary positional embedding with caching."""

  def __init__(
    self,
    dim: int,
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    use_real: bool = False,
    repeat_interleave_real: bool = False,
  ):
    super().__init__()
    assert dim % 2 == 0
    self.dim = dim
    self.theta = theta
    self.theta_rescale_factor = theta_rescale_factor
    self.interpolation_factor = interpolation_factor
    # dtype of freqs
    self.dtype = dtype
    self.use_real = use_real
    self.repeat_interleave_real = repeat_interleave_real

  def build_freqs(self, device):
    freqs = 1.0 / (
      self.theta
      ** (torch.arange(0, self.dim, 2, dtype=self.dtype, device=device)[: (self.dim // 2)] / self.dim).to(device=device)
    )
    return freqs

  def build_freqs_outer(self, pos: torch.Tensor, device):
    theta = self.theta
    # rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if self.theta_rescale_factor != 1.0:
      theta *= self.theta_rescale_factor ** (self.dim / (self.dim - 2))

    freqs = self.build_freqs(device)

    freqs = torch.outer(pos * self.interpolation_factor, freqs)
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()

    if self.use_real and self.repeat_interleave_real:
      freqs_cos = freqs_cos.repeat_interleave(2, dim=1)
      freqs_sin = freqs_sin.repeat_interleave(2, dim=1)

    return freqs_cos.float(), freqs_sin.float()

  @functools.lru_cache(maxsize=16)
  def forward_from_grid(self, seq_len: int, start_pos: int, device_str: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)
    pos = torch.arange(start_pos, start_pos + seq_len, dtype=self.dtype, device=device)

    freqs_cos, freqs_sin = self.build_freqs_outer(pos, device)
    return freqs_cos, freqs_sin


class NDRotaryEmbedding(nn.Module):
  def __init__(
    self,
    rope_dim_list: list[int],
    rope_theta: float,
    theta_rescale_factor: float | list[float] = 1.0,
    interpolation_factor: float | list[float] = 1.0,
    use_real: bool = False,
    repeat_interleave_real: bool = False,
    dtype: torch.dtype = torch.float32,
  ):
    super().__init__()

    self.rope_dim_list = rope_dim_list
    self.ndim = len(rope_dim_list)
    self.rope_theta = rope_theta
    self.dtype = dtype

    if isinstance(theta_rescale_factor, (int, float)):
      self.theta_rescale_factor = [theta_rescale_factor] * self.ndim
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
      self.theta_rescale_factor = [theta_rescale_factor[0]] * self.ndim
    else:
      self.theta_rescale_factor = theta_rescale_factor
    assert len(self.theta_rescale_factor) == self.ndim, "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, (int, float)):
      self.interpolation_factor = [interpolation_factor] * self.ndim
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
      self.interpolation_factor = [interpolation_factor[0]] * self.ndim
    else:
      self.interpolation_factor = interpolation_factor
    assert len(self.interpolation_factor) == self.ndim, "len(interpolation_factor) should equal to len(rope_dim_list)"

    self.rope_generators = nn.ModuleList()
    _config_to_gen_idx: dict[tuple, int] = {}
    self.dim_idx_to_gen_idx: list[int] = []

    for i in range(self.ndim):
      dim = self.rope_dim_list[i]
      rescale = self.theta_rescale_factor[i]
      interp = self.interpolation_factor[i]

      config_key = (dim, rescale, interp, use_real, repeat_interleave_real)
      if config_key not in _config_to_gen_idx:
        generator = OneDRotaryEmbedding(
          dim=dim,
          theta=self.rope_theta,
          theta_rescale_factor=rescale,
          interpolation_factor=interp,
          dtype=self.dtype,
          use_real=use_real,
          repeat_interleave_real=repeat_interleave_real,
        )
        _config_to_gen_idx[config_key] = len(self.rope_generators)
        self.rope_generators.append(generator)

      gen_idx = _config_to_gen_idx[config_key]
      self.dim_idx_to_gen_idx.append(gen_idx)

  def forward_from_grid(
    self,
    grid_size: tuple[int, ...],
    start_frame: int = 0,
    device: torch.device | str | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self._forward_cached_from_grid(grid_size, start_frame, device)

  @functools.lru_cache(maxsize=16)
  def _forward_cached_from_grid(
    self,
    grid_size: tuple[int, ...],
    start_frame: int,
    device_str: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_str)
    sizes = _to_tuple(grid_size, dim=self.ndim)

    # pre allocate outputs to avoid CPU ops and extra cats
    num_tokens = 1
    for s in sizes:
      num_tokens *= int(s)
    head_dim_half = sum(self.rope_dim_list) // 2
    cos = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)
    sin = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)

    # compute per axis 1d embeddings once and expand via repeats to [N, d_i/2]
    col_offset = 0
    for i in range(self.ndim):
      dim_i = self.rope_dim_list[i]
      dim_i_half = dim_i // 2
      size_i = int(sizes[i])

      # Starting position for this axis; only the time axis (i==0) gets a frame offset
      base_offset = start_frame if (i == 0 and start_frame > 0) else 0

      gen_idx = self.dim_idx_to_gen_idx[i]
      generator = self.rope_generators[gen_idx]
      cos_1d, sin_1d = generator.forward_from_grid(size_i, base_offset, device_str)

      # Expand to [num_tokens, dim_i/2] matching flatten order (last dims vary fastest)
      repeats_per_entry = 1
      for j in range(i + 1, self.ndim):
        repeats_per_entry *= int(sizes[j])
      tile_count = 1
      for j in range(i):
        tile_count *= int(sizes[j])

      cos_expanded = cos_1d.repeat_interleave(repeats_per_entry, dim=0)
      sin_expanded = sin_1d.repeat_interleave(repeats_per_entry, dim=0)
      if tile_count > 1:
        cos_expanded = cos_expanded.repeat(tile_count, 1)
        sin_expanded = sin_expanded.repeat(tile_count, 1)

      cos[:, col_offset : col_offset + dim_i_half] = cos_expanded
      sin[:, col_offset : col_offset + dim_i_half] = sin_expanded
      col_offset += dim_i_half

    return cos.float(), sin.float()
