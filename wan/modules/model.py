import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from safetensors.torch import load_file as safetensors_load_file

from wan.configs.models.dits.wan import WanConfig
from wan.layers.attention.layer import USPAttention
from wan.layers.elementwise import MulAdd
from wan.layers.layernorm import LayerNormScaleShift, RMSNorm, ScaleResidualLayerNormScaleShift
from wan.layers.mlp import MLP
from wan.layers.mrope import NDRotaryEmbedding
from wan.layers.rotary_embedding.utils import apply_flashinfer_rope_qk_inplace
from wan.layers.visual_embedding import ModulateProjection, PatchEmbed, TimestepEmbedder
from wan.loader.utils import get_param_names_mapping
from wan.platform import CudaPlatform
from wan.server_args import ServerArgs

__all__ = ['WanModel']


class WanCrossAttention(nn.Module):
  def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6):
    assert dim % num_heads == 0
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)
    self.to_out = nn.Linear(dim, dim)
    self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    self.attn = USPAttention(num_heads=num_heads, head_size=self.head_dim, causal=False)

  def forward(self, x, context):
    q = self.norm_q(self.to_q(x)).unflatten(2, (self.num_heads, self.head_dim))
    k = self.norm_k(self.to_k(context)).unflatten(2, (self.num_heads, self.head_dim))
    v = self.to_v(context).unflatten(2, (self.num_heads, self.head_dim))

    x = self.attn(q, k, v).flatten(2)
    x = self.to_out(x)
    return x


class WanTransformerBlock(nn.Module):
  def __init__(
    self,
    dim,
    ffn_dim: int,
    num_heads: int,
    qk_norm: str = "rms_norm_across_heads",
    eps: float = 1e-6,
  ):
    super().__init__()
    assert qk_norm == "rms_norm_across_heads", f"Unsupported qk_norm: {qk_norm}"

    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    # 1. self attention
    self.norm1 = LayerNormScaleShift(dim, eps=eps, elementwise_affine=False, dtype=torch.float32)
    self.to_q = nn.Linear(dim, dim, bias=True)
    self.to_k = nn.Linear(dim, dim, bias=True)
    self.to_v = nn.Linear(dim, dim, bias=True)
    self.to_out = nn.Linear(dim, dim, bias=True)
    self.attn1 = USPAttention(num_heads=num_heads, head_size=self.head_dim, causal=False)
    self.norm_q = RMSNorm(dim, eps=eps)
    self.norm_k = RMSNorm(dim, eps=eps)
    self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
      dim, eps=eps, elementwise_affine=True, dtype=torch.float32
    )

    # 2. cross attention
    self.attn2 = WanCrossAttention(dim, num_heads, qk_norm=True, eps=eps)
    self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
      dim, eps=eps, elementwise_affine=False, dtype=torch.float32
    )

    # 3. feed forward
    self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
    self.mlp_residual = MulAdd()

    self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

  def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
  ) -> torch.Tensor:
    orig_dtype = hidden_states.dtype

    if temb.dim() == 4:
      shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table.unsqueeze(0) + temb.float()
      ).chunk(6, dim=2)
      shift_msa = shift_msa.squeeze(2)
      scale_msa = scale_msa.squeeze(2)
      gate_msa = gate_msa.squeeze(2)
      c_shift_msa = c_shift_msa.squeeze(2)
      c_scale_msa = c_scale_msa.squeeze(2)
      c_gate_msa = c_gate_msa.squeeze(2)
    else:
      e = self.scale_shift_table + temb.float()
      shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=1)

    # 1. self attention
    norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa)
    query = self.norm_q(self.to_q(norm_hidden_states)).unflatten(2, (self.num_heads, self.head_dim))
    key = self.norm_k(self.to_k(norm_hidden_states)).unflatten(2, (self.num_heads, self.head_dim))
    value = self.to_v(norm_hidden_states).unflatten(2, (self.num_heads, self.head_dim))

    cos, sin = freqs_cis
    cos_sin_cache = torch.cat([cos.contiguous(), sin.contiguous()], dim=-1)
    query, key = apply_flashinfer_rope_qk_inplace(query, key, cos_sin_cache, is_neox=False)

    attn_output = self.to_out(self.attn1(query, key, value).flatten(2))

    null_shift = null_scale = torch.zeros((1,), device=hidden_states.device, dtype=hidden_states.dtype)
    norm_hidden_states, hidden_states = self.self_attn_residual_norm(
      hidden_states, attn_output, gate_msa, null_shift, null_scale
    )
    norm_hidden_states, hidden_states = norm_hidden_states.to(orig_dtype), hidden_states.to(orig_dtype)

    # 2. cross attention
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states)
    norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
      hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
    )
    norm_hidden_states, hidden_states = norm_hidden_states.to(orig_dtype), hidden_states.to(orig_dtype)

    # 3. feed forward
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = self.mlp_residual(ff_output, c_gate_msa, hidden_states).to(orig_dtype)
    return hidden_states


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
    timestep_proj = self.time_modulation(temb).unflatten(-1, (6, -1))

    encoder_hidden_states_text = self.text_embedder(encoder_hidden_states_text)

    return temb, timestep_proj, encoder_hidden_states_text


class WanModel(ModelMixin, ConfigMixin):
  """Wan diffusion backbone supporting both text-to-video and image-to-video."""

  def __init__(
    self,
    config: WanConfig,
  ):
    super().__init__()

    self.patch_size = config.arch_config.patch_size

    inner_dim = config.arch_config.num_attention_heads * config.arch_config.attention_head_dim
    self.hidden_size = config.arch_config.hidden_size

    # since kernel_size = patch_size = stride, we can use PatchEmbed instead of nn.Conv3d
    self.patch_embedding = PatchEmbed(
      in_chans=config.arch_config.in_dim,
      embed_dim=inner_dim,
      patch_size=config.arch_config.patch_size,
      flatten=False,
    )

    self.condition_embedder = WanTimeTextImageEmbedding(
      dim=inner_dim, time_freq_dim=config.arch_config.freq_dim, text_embed_dim=config.arch_config.text_dim
    )

    self.blocks = nn.ModuleList(
      [
        WanTransformerBlock(
          dim=inner_dim,
          ffn_dim=config.arch_config.ffn_dim,
          num_heads=config.arch_config.num_attention_heads,
          qk_norm=config.arch_config.qk_norm,
          eps=config.arch_config.eps,
        )
        for _ in range(config.arch_config.num_layers)
      ]
    )

    self.norm_out = LayerNormScaleShift(
      inner_dim, eps=config.arch_config.eps, elementwise_affine=False, dtype=torch.float32
    )
    self.proj_out = nn.Linear(
      inner_dim, config.arch_config.num_channels_latents * math.prod(self.patch_size), bias=True
    )
    self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    d = self.hidden_size // config.arch_config.num_attention_heads
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    self.rotary_emb = NDRotaryEmbedding(rope_dim_list=rope_dim_list, rope_theta=10000, dtype=torch.float64)

  def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: list[torch.Tensor],
  ):
    """
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

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    freqs_cis = self.rotary_emb.forward_from_grid(
      (post_patch_num_frames, post_patch_height, post_patch_width),
      start_frame=0,
      device=hidden_states.device,
    )

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

  def load(self, model_path: str, server_args: ServerArgs):
    gpu_mem_before_loading = CudaPlatform.get_available_gpu_memory()
    print(f"Loading Transformer from {model_path}. avail mem: {gpu_mem_before_loading:.2f} GB")
    state_dict = safetensors_load_file(model_path)

    arch = server_args.pipeline_config.dit_config.arch_config

    mapping_fn = get_param_names_mapping(arch.param_names_mapping)
    state_dict = {mapping_fn(k)[0]: v for k, v in state_dict.items()}

    self.load_state_dict(state_dict, strict=True)
    self.eval().requires_grad_(False)
