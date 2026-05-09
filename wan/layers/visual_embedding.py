import math

import torch
import torch.nn.functional as F
from torch import nn

from wan.layers.activation import get_act_fn
from wan.layers.mlp import MLP


def timestep_embedding(
  t: torch.Tensor,
  dim: int,
  max_period: int = 10000,
  dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """
  Create sinusoidal timestep embeddings.

  Args:
    t: Tensor of shape [B] with timesteps
    dim: Embedding dimension
    max_period: Controls the minimum frequency of the embeddings

  Returns:
    Tensor of shape [B, dim] with embeddings
  """
  half = dim // 2
  freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype, device=t.device) / half)
  args = t[:, None].float() * freqs[None]
  embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
  if dim % 2:
    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
  return embedding


class TimestepEmbedder(nn.Module):
  def __init__(
    self,
    hidden_size,
    act_layer="silu",
    frequency_embedding_size=256,
    max_period=10000,
    dtype=None,
    freq_dtype=torch.float32,
  ):
    super().__init__()
    self.frequency_embedding_size = frequency_embedding_size
    self.max_period = max_period

    self.mlp = MLP(
      frequency_embedding_size,
      hidden_size,
      hidden_size,
      act_type=act_layer,
    )
    self.freq_dtype = freq_dtype

  def forward(
    self,
    t: torch.Tensor,
    timestep_seq_len: int | None = None,
  ) -> torch.Tensor:
    t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period, dtype=self.freq_dtype).to(
      self.mlp.fc_in.weight.dtype
    )

    if timestep_seq_len is not None:
      assert (t_freq.shape[0] % timestep_seq_len) == 0
      batch_size = t_freq.shape[0] // timestep_seq_len
      t_freq = t_freq.unflatten(0, (batch_size, timestep_seq_len))

    t_emb = self.mlp(t_freq)
    return t_emb


class PatchEmbed(nn.Module):
  def __init__(
    self,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    norm_layer=None,
    flatten=True,
    bias=True,
    dtype=None,
    prefix: str = "",
  ):
    super().__init__()
    if isinstance(patch_size, list | tuple):
      if len(patch_size) == 1:
        patch_size = (1, patch_size[0], patch_size[0])
      elif len(patch_size) == 2:
        patch_size = (1, patch_size[0], patch_size[1])
    else:
      patch_size = (1, patch_size, patch_size)

    self.patch_size = patch_size
    self.flatten = flatten

    self.proj = nn.Conv3d(
      in_chans,
      embed_dim,
      kernel_size=patch_size,
      stride=patch_size,
      bias=bias,
      dtype=dtype,
    )
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    if x.dim() == 5:
      B, C, T, H, W = x.shape
      pt, ph, pw = self.patch_size

      if T % pt == 0 and H % ph == 0 and W % pw == 0:
        T_ = T // pt
        H_ = H // ph
        W_ = W // pw

        x = x.reshape(B, C, T_, pt, H_, ph, W_, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, T_ * H_ * W_, C * pt * ph * pw)

        w = self.proj.weight.reshape(self.proj.weight.shape[0], -1)
        x = F.linear(x, w, self.proj.bias)  # [B, T'*H'*W', embed_dim]

        if not self.flatten:
          x = x.reshape(B, T_, H_, W_, -1).permute(0, 4, 1, 2, 3).contiguous()

        x = self.norm(x)
        return x

    # Fallback to Conv3d for non-5D input or indivisible spatial dims.
    x = self.proj(x)
    if self.flatten:
      x = x.flatten(2).transpose(1, 2)
    x = self.norm(x)
    return x


class ModulateProjection(nn.Module):
  def __init__(
    self,
    hidden_size: int,
    factor: int = 2,
    act_layer: str = "silu",
    dtype: torch.dtype | None = None,
  ):
    super().__init__()
    self.factor = factor
    self.hidden_size = hidden_size
    self.linear = nn.Linear(
      hidden_size,
      hidden_size * factor,
      bias=True,
    )
    self.act = get_act_fn(act_layer)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.act(x)
    x = self.linear(x)
    return x
