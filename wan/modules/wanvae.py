import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file as safetensors_load_file

from wan.configs.models.vaes.wanvae import WanVAEConfig
from wan.platform import CudaPlatform, get_local_torch_device
from wan.server_args import ServerArgs

CACHE_T = 2


class DiagonalGaussianDistribution:
  def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)
    if self.deterministic:
      self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

  def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
    # make sure sample is on the same device as the parameters and has same dtype
    sample = randn_tensor(
      self.mean.shape,
      generator=generator,
      device=self.parameters.device,
      dtype=self.parameters.dtype,
    )
    x = self.mean + self.std * sample
    return x

  def kl(
    self,
    other: "DiagonalGaussianDistribution | None" = None,
    dims: tuple[int, ...] = (1, 2, 3),
  ) -> torch.Tensor:
    if self.deterministic:
      return torch.Tensor([0.0])
    else:
      if other is None:
        return 0.5 * torch.sum(
          torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
          dim=dims,
        )
      else:
        return 0.5 * torch.sum(
          torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
          dim=dims,
        )

  def nll(self, sample: torch.Tensor, dims: tuple[int, ...] = (1, 2, 3)) -> torch.Tensor:
    if self.deterministic:
      return torch.Tensor([0.0])
    logtwopi = np.log(2.0 * np.pi)
    return 0.5 * torch.sum(
      logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
      dim=dims,
    )

  def mode(self) -> torch.Tensor:
    return self.mean


class CausalConv3d(nn.Conv3d):
  """
  Causal 3d convolusion.
  """

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
  ):
    super().__init__(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
    )
    self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
    self.padding = (0, 0, 0)

  def forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
      cache_x = cache_x.to(x.device)
      x = torch.cat([cache_x, x], dim=2)
      padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)

    return super().forward(x)


class RMS_norm(nn.Module):
  def __init__(self, dim, channel_first=True, images=True, bias=False):
    super().__init__()
    broadcastable_dims = (1, 1, 1) if not images else (1, 1)
    shape = (dim, *broadcastable_dims) if channel_first else (dim,)

    self.channel_first = channel_first
    self.scale = dim**0.5
    self.gamma = nn.Parameter(torch.ones(shape))
    self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

  def forward(self, x):
    return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


def _to_vae_channels_last(model):
  for module in model.modules():
    if isinstance(module, nn.Conv2d):
      module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)
    elif isinstance(module, nn.Conv3d):
      module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
  return model


class Upsample(nn.Upsample):
  def forward(self, x):
    """
    Fix bfloat16 support for nearest neighbor interpolation.
    """
    return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
  def __init__(self, dim, mode):
    assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d', 'downsample3d')
    super().__init__()
    self.dim = dim
    self.mode = mode

    # layers
    if mode == 'upsample2d':
      self.resample = nn.Sequential(
        Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, dim // 2, 3, padding=1)
      )
    elif mode == 'upsample3d':
      self.resample = nn.Sequential(
        Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, dim // 2, 3, padding=1)
      )
      self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

    elif mode == 'downsample2d':
      self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
    elif mode == 'downsample3d':
      self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
      self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    else:
      self.resample = nn.Identity()

  def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
    b, c, t, h, w = x.size()

    if self.mode == 'upsample3d':
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = 'Rep'
          feat_idx[0] += 1
        else:
          cache_x = x[:, :, -CACHE_T:, :, :]
          if feat_cache[idx] == 'Rep':
            x = self.time_conv(x)
          else:
            x = self.time_conv(x, feat_cache[idx])

          feat_cache[idx] = cache_x
          feat_idx[0] += 1

          x = x.reshape(b, 2, c, t, h, w)
          x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
          x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous(memory_format=torch.channels_last)
    x = self.resample(x)
    x = x.unflatten(0, (-1, t)).permute(0, 2, 1, 3, 4).contiguous(memory_format=torch.channels_last_3d)

    if self.mode == 'downsample3d':
      if feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
          feat_cache[idx] = x
        else:
          cache_x = x[:, :, -1:, :, :]
          x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
          feat_cache[idx] = cache_x

          deferred_x = feat_cache[idx + 1]
          if deferred_x is not None:
            x = torch.cat([deferred_x, x], 2)
            feat_cache[idx + 1] = None

          if x.shape[2] == 1 and not final:
            feat_cache[idx + 1] = x
            x = None

        feat_idx[0] += 2
    return x


class ResidualBlock(nn.Module):
  def __init__(self, in_dim: int, out_dim: int, dropout=0.0):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim

    self.residual = nn.Sequential(
      RMS_norm(in_dim, images=False),
      nn.SiLU(),
      CausalConv3d(in_dim, out_dim, 3, padding=1),
      RMS_norm(out_dim, images=False),
      nn.SiLU(),
      nn.Dropout(dropout),
      CausalConv3d(out_dim, out_dim, 3, padding=1),
    )
    self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

  def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
    # apply shortcut connection
    h = self.shortcut(x)

    x = self.residual[0](x)
    x = self.residual[1](x)

    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
      # cache last frame of last two chunk
      cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    x = self.residual[2](x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1

    x = self.residual[3](x)
    x = self.residual[4](x)

    x = self.residual[5](x)

    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
      # cache last frame of last two chunk
      cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
    x = self.residual[6](x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1

    return x + h


class AttentionBlock(nn.Module):
  """
  Causal self-attention with a single head.
  """

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

    # layers
    self.norm = RMS_norm(dim)
    self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
    self.proj = nn.Conv2d(dim, dim, 1)

  def forward(self, x):
    identity = x
    b, c, t, h, w = x.size()
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.norm(x)

    # compute query, key, value
    q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

    # apply attention
    x = F.scaled_dot_product_attention(
      q,
      k,
      v,
    )
    x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

    # output
    x = self.proj(x)
    x = x.view(b, t, c, h, w)
    x = x.permute(0, 2, 1, 3, 4)
    return x + identity


class WanEncoder3d(nn.Module):
  def __init__(
    self,
    dim=128,
    z_dim=4,
    dim_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attn_scales=(),
    temperal_downsample=(True, True, False),
    dropout=0.0,
  ):
    super().__init__()
    self.dim = dim
    self.z_dim = z_dim
    dim_mult = list(dim_mult)
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_downsample = temperal_downsample

    # dimensions
    dims = [dim * u for u in [1] + dim_mult]
    scale = 1.0

    # init block
    self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

    # downsample blocks
    self.downsamples = nn.ModuleList([])

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      # residual (+attention) blocks
      for _ in range(num_res_blocks):
        self.downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
        if scale in attn_scales:
          self.downsamples.append(AttentionBlock(out_dim))
        in_dim = out_dim

      # downsample block
      if i != len(dim_mult) - 1:
        mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
        self.downsamples.append(Resample(out_dim, mode=mode))
        scale /= 2.0

    # middle blocks
    self.middle = nn.Sequential(
      ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout)
    )

    self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

  def forward(self, x, feat_cache=None, feat_idx=[0], final=False):
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = x[:, :, -CACHE_T:, :, :]
      x = self.conv1(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv1(x)

    # downsamples
    for layer in self.downsamples:
      if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx, final=final)
        if x is None:
          return None
      else:
        x = layer(x)

    # middle
    for layer in self.middle:
      if isinstance(layer, ResidualBlock) and feat_cache is not None:
        x = layer(x, feat_cache, feat_idx, final=final)
      else:
        x = layer(x)

    # head
    for layer in self.head:
      if isinstance(layer, CausalConv3d) and feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :]
        x = layer(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
      else:
        x = layer(x)
    return x


class Decoder3d(nn.Module):
  def __init__(
    self,
    dim=128,
    z_dim=4,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temperal_upsample=[False, True, True],
    dropout=0.0,
  ):
    super().__init__()
    dim_mult = list(dim_mult)
    self.dim = dim
    self.z_dim = z_dim
    self.dim_mult = dim_mult
    self.num_res_blocks = num_res_blocks
    self.attn_scales = attn_scales
    self.temperal_upsample = temperal_upsample

    # dimensions
    dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    scale = 1.0 / 2 ** (len(dim_mult) - 2)

    # init block
    self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

    # middle blocks
    self.middle = nn.Sequential(
      ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout)
    )

    # upsample blocks
    upsamples = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      # residual (+attention) blocks
      if i == 1 or i == 2 or i == 3:
        in_dim = in_dim // 2
      for _ in range(num_res_blocks + 1):
        upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
        if scale in attn_scales:
          upsamples.append(AttentionBlock(out_dim))
        in_dim = out_dim

      # upsample block
      if i != len(dim_mult) - 1:
        mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
        upsamples.append(Resample(out_dim, mode=mode))
        scale *= 2.0
    self.upsamples = nn.Sequential(*upsamples)

    self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

  def forward(self, x, feat_cache=None, feat_idx=[0]):
    # conv1
    if feat_cache is not None:
      idx = feat_idx[0]
      cache_x = x[:, :, -CACHE_T:, :, :].clone()
      if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        # cache last frame of last two chunk
        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
      x = self.conv1(x, feat_cache[idx])
      feat_cache[idx] = cache_x
      feat_idx[0] += 1
    else:
      x = self.conv1(x)

    # middle
    for layer in self.middle:
      if isinstance(layer, ResidualBlock) and feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
      else:
        x = layer(x)

    # upsamples
    for layer in self.upsamples:
      if feat_cache is not None:
        x = layer(x, feat_cache, feat_idx)
      else:
        x = layer(x)

    # head
    for layer in self.head:
      if isinstance(layer, CausalConv3d) and feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
          # cache last frame of last two chunk
          cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
        x = layer(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
      else:
        x = layer(x)
    return x


def count_conv3d(model):
  count = 0
  for m in model.modules():
    if isinstance(m, CausalConv3d):
      count += 1
  return count


def count_cache_layers(model):
  count = 0
  for m in model.modules():
    if isinstance(m, CausalConv3d) or (isinstance(m, Resample) and m.mode == 'downsample3d'):
      count += 1
  return count


class Wan2_1_VAE(nn.Module):
  def __init__(self, config: WanVAEConfig):
    super().__init__()
    self.z_dim = config.z_dim
    self.latents_mean = list(config.latents_mean)
    self.latents_std = list(config.latents_std)

    self.encoder = WanEncoder3d(
      config.base_dim,
      self.z_dim * 2,
      config.dim_mult,
      config.num_res_blocks,
      config.attn_scales,
      config.temperal_downsample,
      config.dropout,
    )
    self.conv1 = CausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
    self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)
    self.decoder = Decoder3d(
      config.base_dim,
      self.z_dim,
      config.dim_mult,
      config.num_res_blocks,
      config.attn_scales,
      config.temperal_downsample[::-1],
      config.dropout,
    )

  def load(self, model_path: str, server_args: ServerArgs):
    gpu_mem_before_loading = CudaPlatform.get_available_gpu_memory()
    print(f"Loading VAE from {model_path}. avail mem: {gpu_mem_before_loading:.2f} GB")

    t0 = time.perf_counter()
    target_device = get_local_torch_device()
    state_dict = safetensors_load_file(model_path, device=str(target_device))
    t_read = time.perf_counter() - t0

    t1 = time.perf_counter()
    self.load_state_dict(state_dict, strict=True, assign=True)
    t_copy = time.perf_counter() - t1

    self.eval().requires_grad_(False)
    _to_vae_channels_last(self)
    print(f"  VAE load: read={t_read:.2f}s  load_state_dict={t_copy:.2f}s  total={(t_read + t_copy):.2f}s")

  def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
    dtype = next(self.parameters()).dtype
    with torch.amp.autocast("cuda", dtype=dtype):
      self.clear_cache()
      t = x.shape[2]
      t = 1 + ((t - 1) // 4) * 4
      iter_ = 1 + (t - 1) // 2
      feat_map = [None] * count_cache_layers(self.encoder) if iter_ > 1 else None

      # 对encode输入的x，按时间拆分为1、4、4、4....
      for i in range(iter_):
        conv_idx = [0]
        if i == 0:
          out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_map, feat_idx=conv_idx)
        else:
          out_ = self.encoder(
            x[:, :, 1 + 2 * (i - 1) : 1 + 2 * i, :, :],
            feat_cache=feat_map,
            feat_idx=conv_idx,
            final=(i == (iter_ - 1)),
          )
          if out_ is None:
            continue
          out = torch.cat([out, out_], 2)

      enc = self.conv1(out)
      mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
      return DiagonalGaussianDistribution(torch.cat([mu, logvar], dim=1))

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    dtype = next(self.parameters()).dtype
    with torch.amp.autocast("cuda", dtype=dtype):
      self.clear_cache()
      iter_ = z.shape[2]
      x = self.conv2(z)
      for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
          out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        else:
          out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
          out = torch.cat([out, out_], 2)
      self.clear_cache()
      return out

  def clear_cache(self):
    self._conv_num = count_conv3d(self.decoder)
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num
    self._enc_conv_num = count_conv3d(self.encoder)
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num
