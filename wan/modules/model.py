# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from wan.layers.mlp import MLP
from wan.layers.mrope import NDRotaryEmbedding
from wan.layers.visual_embedding import ModulateProjection, PatchEmbed, TimestepEmbedder

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
  # preprocess
  assert dim % 2 == 0
  half = dim // 2
  position = position.type(torch.float64)

  # calculation
  sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
  x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
  return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
  assert dim % 2 == 0
  freqs = torch.outer(
    torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
  )
  freqs = torch.polar(torch.ones_like(freqs), freqs)
  return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
  n, c = x.size(2), x.size(3) // 2

  # split freqs
  freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

  # loop over samples
  output = []
  for i, (f, h, w) in enumerate(grid_sizes.tolist()):
    seq_len = f * h * w

    # precompute multipliers
    x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
    freqs_i = torch.cat(
      [
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
      ],
      dim=-1,
    ).reshape(seq_len, 1, -1)

    # apply rotary embedding
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[i, seq_len:]])

    # append to collection
    output.append(x_i)
  return torch.stack(output).float()


class WanRMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
    """
    return self._norm(x.float()).type_as(x) * self.weight

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
  def __init__(self, dim, eps=1e-6, elementwise_affine=False):
    super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

  def forward(self, x):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
    """
    return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
  def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
    assert dim % num_heads == 0
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.eps = eps

    # layers
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.o = nn.Linear(dim, dim)
    self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

  def forward(self, x, seq_lens, grid_sizes, freqs):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    # query, key, value function
    def qkv_fn(x):
      q = self.norm_q(self.q(x)).view(b, s, n, d)
      k = self.norm_k(self.k(x)).view(b, s, n, d)
      v = self.v(x).view(b, s, n, d)
      return q, k, v

    q, k, v = qkv_fn(x)

    x = flash_attention(
      q=rope_apply(q, grid_sizes, freqs),
      k=rope_apply(k, grid_sizes, freqs),
      v=v,
      k_lens=seq_lens,
      window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class WanCrossAttention(WanSelfAttention):
  def forward(self, x, context, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L1, C]
        context(Tensor): Shape [B, L2, C]
        context_lens(Tensor): Shape [B]
    """
    b, n, d = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value
    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)

    # compute attention
    x = flash_attention(q, k, v, k_lens=context_lens)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class WanTransformerBlock(nn.Module):
  def __init__(
    self,
    dim,
    ffn_dim: int,
    num_heads: int,
  ):
    super().__init__()

    # 1. self attention

    # todo: implemnt kernel for this
    self.norm1 = None


class WanAttentionBlock(nn.Module):
  def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
    super().__init__()

    self.dim = dim
    self.ffn_dim = ffn_dim
    self.num_heads = num_heads
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.cross_attn_norm = cross_attn_norm
    self.eps = eps

    # layers
    self.norm1 = WanLayerNorm(dim, eps)
    self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
    self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
    self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
    self.norm2 = WanLayerNorm(dim, eps)
    self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))

    # modulation
    self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

  def forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
  ):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, L1, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    assert e.dtype == torch.float32
    with torch.amp.autocast('cuda', dtype=torch.float32):
      e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    # self-attention
    y = self.self_attn(self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), seq_lens, grid_sizes, freqs)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      x = x + y * e[2].squeeze(2)

    # cross-attention & ffn function
    def cross_attn_ffn(x, context, context_lens, e):
      x = x + self.cross_attn(self.norm3(x), context, context_lens)
      y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
      with torch.amp.autocast('cuda', dtype=torch.float32):
        x = x + y * e[5].squeeze(2)
      return x

    x = cross_attn_ffn(x, context, context_lens, e)
    return x


class WanTimeTextImageEmbedding(nn.Module):
  def __init__(self, dim: int, time_freq_dim: int, text_embed_dim: int):
    super().__init__()

    self.time_embedder = TimestepEmbedder(
      dim,
      frequency_embedding_size=time_freq_dim,
      act_layer="silu",
    )

    self.time_modulation = ModulateProjection(
      dim,
      factor=6,
      act_layer="silu",
    )

    self.text_embedder = MLP(
      text_embed_dim,
      dim,
      dim,
      act_type="gelu_pytorch_tanh",
    )

  def forward(
    self,
    timestep: torch.Tensor,
    encoder_hidden_states_text: torch.Tensor,
    timestep_seq_len: int | None = None,
  ):
    temb = self.time_embedder(timestep, timestep_seq_len)
    timestep_proj = self.time_modulation(temb)

    encoder_hidden_states_text = self.text_embedder(encoder_hidden_states_text)

    return temb, timestep_proj, encoder_hidden_states_text


class WanModel(ModelMixin, ConfigMixin):
  r"""
  Wan diffusion backbone supporting both text-to-video and image-to-video.
  """

  ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size']
  _no_split_modules = ['WanAttentionBlock']

  @register_to_config
  def __init__(
    self,
    model_type='t2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,
    dim=2048,
    ffn_dim=8192,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=16,
    num_layers=32,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
  ):
    r"""
    Initialize the diffusion model backbone.

    Args:
      model_type (`str`, *optional*, defaults to 't2v'):
        Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
      patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
        3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
      text_len (`int`, *optional*, defaults to 512):
        Fixed length for text embeddings
      in_dim (`int`, *optional*, defaults to 16):
        Input video channels (C_in)
      dim (`int`, *optional*, defaults to 2048):
        Hidden dimension of the transformer
      ffn_dim (`int`, *optional*, defaults to 8192):
        Intermediate dimension in feed-forward network
      freq_dim (`int`, *optional*, defaults to 256):
        Dimension for sinusoidal time embeddings
      text_dim (`int`, *optional*, defaults to 4096):
        Input dimension for text embeddings
      out_dim (`int`, *optional*, defaults to 16):
        Output video channels (C_out)
      num_heads (`int`, *optional*, defaults to 16):
        Number of attention heads
      num_layers (`int`, *optional*, defaults to 32):
        Number of transformer blocks
      window_size (`tuple`, *optional*, defaults to (-1, -1)):
        Window size for local attention (-1 indicates global attention)
      qk_norm (`bool`, *optional*, defaults to True):
        Enable query/key normalization
      cross_attn_norm (`bool`, *optional*, defaults to False):
        Enable cross-attention normalization
      eps (`float`, *optional*, defaults to 1e-6):
        Epsilon value for normalization layers
    """

    super().__init__()

    self.patch_size = patch_size
    self.text_len = text_len
    self.in_dim = in_dim
    self.dim = dim
    self.ffn_dim = ffn_dim
    self.freq_dim = freq_dim
    self.text_dim = text_dim
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.window_size = window_size
    self.qk_norm = qk_norm
    self.cross_attn_norm = cross_attn_norm
    self.eps = eps

    # patch & position embedding
    # since kernel_size = patch_size = stride, we can use PatchEmbed instead of nn.Conv3d
    self.patch_embedding = PatchEmbed(in_chans=in_dim, embed_dim=dim, patch_size=patch_size, flatten=False)

    self.condition_embedder = WanTimeTextImageEmbedding(
      dim=dim,
      time_freq_dim=freq_dim,
      text_embed_dim=text_dim,
    )

    # self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
    # self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    # self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    # blocks
    self.blocks = nn.ModuleList(
      [
        WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        for _ in range(num_layers)
      ]
    )

    # output norm & projection
    self.norm_out = WanLayerNorm(dim, eps)

    self.proj_out = nn.Linear(dim, out_dim * math.prod(patch_size), bias=True)
    self.scale_shift_table = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    # get rotary embeddings
    d = dim // num_heads
    self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]

    self.rotary_emb = NDRotaryEmbedding(rope_dim_list=self.rope_dim_list, rope_theta=10000, dtype=torch.float64)

    self.freqs = torch.cat(
      [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: list[torch.Tensor],
  ):
    r"""
    Forward pass through the diffusion model

    Args:
      hidden_states (Tensor):
        Full model input tensor of shape [B, C_in, F, H, W]
      timestep (Tensor):
        Diffusion timesteps tensor of shape [B, seq_len]. The sequence length
        used for positional encoding / padding is derived from this shape.
      encoder_hidden_states (List[Tensor]):
        List of text embeddings each with shape [L, C]

    Returns:
      Tensor:
        Denoised video tensor of shape [B, C_out, F, H / 8, W / 8]
    """
    orig_dtype = hidden_states.dtype

    batch_size, num_channels, num_frames, height, width = hidden_states.shape

    p_t, p_h, p_w = self.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # rotary emb
    freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
      (
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
      ),
      start_frame=0,
      device=hidden_states.device,
    )

    freqs_cis = (freqs_cos.float(), freqs_sin.float()) if freqs_cos is not None else None

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    if timestep.dim() == 2:
      ts_seq_len = timestep.shape[1]
      timestep = timestep.flatten()
    else:
      ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
      timestep, encoder_hidden_states, timestep_seq_len=ts_seq_len
    )

    assert encoder_hidden_states.dtype == orig_dtype

    # transformer blocks
    for block in self.blocks:
      hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis)

    # output norm, projection & unpatchify
    if temb.dim() == 3:
      # batch_size, seq_len, dim
      shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
      shift = shift.squeeze(2)
      scale = scale.squeeze(2)
    else:
      # batch_size, dim
      shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    hidden_states = self.norm_out(hidden_states, shift, scale)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
      batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )

    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    return output
