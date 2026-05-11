# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm

from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

_LOW_NOISE_I2V_CHECKPOINT = "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
_HIGH_NOISE_I2V_CHECKPOINT = "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"


_COMFYUI_VAE_CHECKPOINTS = {
  "Wan2.1_VAE.pth": (
    "wan_2.1_vae.safetensors",
    "Wan2_1_VAE_bf16.safetensors",
  ),
}


def _first_existing_path(paths):
  paths = list(paths)
  for path in paths:
    if os.path.exists(path):
      return path
  return paths[0]


def _resolve_comfyui_file(checkpoint_dir, filename, subfolder, alternates=()):
  candidates = [os.path.join(checkpoint_dir, filename)]
  candidates.extend(os.path.join(checkpoint_dir, subfolder, candidate) for candidate in (filename, *alternates))
  return _first_existing_path(candidates)


def _resolve_t5_tokenizer_path(checkpoint_dir, tokenizer_name):
  local_path = os.path.join(checkpoint_dir, tokenizer_name)
  if os.path.exists(local_path):
    return local_path
  return tokenizer_name


def _create_i2v_wan_model(config):
  with init_empty_weights():
    return WanModel(
      patch_size=config.patch_size,
      in_dim=getattr(config, "in_dim", 36),
      dim=config.dim,
      ffn_dim=config.ffn_dim,
      freq_dim=config.freq_dim,
      out_dim=getattr(config, "out_dim", 16),
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      qk_norm=config.qk_norm,
      eps=config.eps,
    )


def _load_i2v_wan_model(checkpoint_path, config):
  if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Could not find I2V diffusion checkpoint: {checkpoint_path}")

  logging.info(f"Loading WanModel from safetensors checkpoint: {checkpoint_path}")
  model = _create_i2v_wan_model(config)
  state_dict = load_safetensors_file(checkpoint_path, device="cpu")
  for k in list(state_dict.keys()):
    state_dict[k] = state_dict.pop(k).to(config.param_dtype)

  if "patch_embedding.weight" in state_dict:
    state_dict["patch_embedding.proj.weight"] = state_dict.pop("patch_embedding.weight")
  if "patch_embedding.bias" in state_dict:
    state_dict["patch_embedding.proj.bias"] = state_dict.pop("patch_embedding.bias")

  key_map = {
    "text_embedding.0.weight": "condition_embedder.text_embedder.fc_in.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.fc_in.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.fc_out.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.fc_out.bias",
    "time_embedding.0.weight": "condition_embedder.time_embedder.mlp.fc_in.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.mlp.fc_in.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.mlp.fc_out.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.mlp.fc_out.bias",
    "time_projection.1.weight": "condition_embedder.time_modulation.linear.weight",
    "time_projection.1.bias": "condition_embedder.time_modulation.linear.bias",
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
    "head.modulation": "scale_shift_table",
  }
  for old_key, new_key in key_map.items():
    if old_key in state_dict:
      state_dict[new_key] = state_dict.pop(old_key)

  # Per-block remap: legacy WanAttentionBlock → WanTransformerBlock
  block_suffix_map = {
    "self_attn.q.weight": "to_q.weight",
    "self_attn.q.bias": "to_q.bias",
    "self_attn.k.weight": "to_k.weight",
    "self_attn.k.bias": "to_k.bias",
    "self_attn.v.weight": "to_v.weight",
    "self_attn.v.bias": "to_v.bias",
    "self_attn.o.weight": "to_out.weight",
    "self_attn.o.bias": "to_out.bias",
    "self_attn.norm_q.weight": "norm_q.weight",
    "self_attn.norm_k.weight": "norm_k.weight",
    "modulation": "scale_shift_table",
    "ffn.0.weight": "ffn.fc_in.weight",
    "ffn.0.bias": "ffn.fc_in.bias",
    "ffn.2.weight": "ffn.fc_out.weight",
    "ffn.2.bias": "ffn.fc_out.bias",
    "norm3.weight": "self_attn_residual_norm.norm.weight",
    "norm3.bias": "self_attn_residual_norm.norm.bias",
    "cross_attn.q.weight": "attn2.to_q.weight",
    "cross_attn.q.bias": "attn2.to_q.bias",
    "cross_attn.k.weight": "attn2.to_k.weight",
    "cross_attn.k.bias": "attn2.to_k.bias",
    "cross_attn.v.weight": "attn2.to_v.weight",
    "cross_attn.v.bias": "attn2.to_v.bias",
    "cross_attn.o.weight": "attn2.to_out.weight",
    "cross_attn.o.bias": "attn2.to_out.bias",
    "cross_attn.norm_q.weight": "attn2.norm_q.weight",
    "cross_attn.norm_k.weight": "attn2.norm_k.weight",
  }
  for i in range(config.num_layers):
    for old_suffix, new_suffix in block_suffix_map.items():
      old_key = f"blocks.{i}.{old_suffix}"
      if old_key in state_dict:
        state_dict[f"blocks.{i}.{new_suffix}"] = state_dict.pop(old_key)

  model.load_state_dict(state_dict, strict=True, assign=True)
  return model


_LORA_BLOCK_MODULE_MAP = {
  "self_attn.q": "to_q",
  "self_attn.k": "to_k",
  "self_attn.v": "to_v",
  "self_attn.o": "to_out",
  "cross_attn.q": "attn2.to_q",
  "cross_attn.k": "attn2.to_k",
  "cross_attn.v": "attn2.to_v",
  "cross_attn.o": "attn2.to_out",
  "ffn.0": "ffn.fc_in",
  "ffn.2": "ffn.fc_out",
}
# kohya/sd-scripts encode the same paths with underscores instead of dots
_LORA_BLOCK_MODULE_MAP_KOHYA = {k.replace(".", "_"): v for k, v in _LORA_BLOCK_MODULE_MAP.items()}


def _remap_lora_module_name(name):
  # diffusion_model.blocks.N.<suffix.with.dots>
  if name.startswith("blocks."):
    rest = name.split(".", 2)
    if len(rest) == 3:
      return f"{rest[0]}.{rest[1]}.{_LORA_BLOCK_MODULE_MAP.get(rest[2], rest[2])}"
  # lora_unet_blocks_N_<suffix_with_underscores>  (already stripped of lora_unet_)
  if name.startswith("blocks_"):
    parts = name.split("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
      suffix = _LORA_BLOCK_MODULE_MAP_KOHYA.get(parts[2], parts[2].replace("_", "."))
      return f"blocks.{parts[1]}.{suffix}"
  return name


def _merge_lora_into_wan_model(model, lora_path, strength=1.0):
  if not os.path.isfile(lora_path):
    raise FileNotFoundError(f"Could not find WanModel LoRA checkpoint: {lora_path}")

  logging.info(f"Merging WanModel LoRA from safetensors checkpoint: {lora_path}")
  state_dict = load_safetensors_file(lora_path, device="cpu")
  down_suffix = ".lora_down.weight"

  for key, down_weight in state_dict.items():
    if not key.endswith(down_suffix):
      continue

    prefix = key[: -len(down_suffix)]
    up_key = f"{prefix}.lora_up.weight"
    alpha_key = f"{prefix}.alpha"
    if up_key not in state_dict:
      raise KeyError(f"Missing LoRA up weight for {key}: expected {up_key}")

    stripped = prefix.removeprefix("diffusion_model.").removeprefix("lora_unet_")
    module_name = _remap_lora_module_name(stripped)
    module = model.get_submodule(module_name)
    if not hasattr(module, "weight"):
      raise TypeError(f"LoRA target module has no weight parameter: {module_name}")

    up_weight = state_dict[up_key]
    rank = down_weight.shape[0]
    alpha = state_dict[alpha_key].item() if alpha_key in state_dict else rank
    scale = strength * float(alpha) / rank
    delta = torch.matmul(up_weight.float(), down_weight.float()) * scale

    if delta.shape != module.weight.shape:
      raise ValueError(
        f"LoRA delta shape {tuple(delta.shape)} does not match {module_name}.weight shape {tuple(module.weight.shape)}"
      )

    with torch.no_grad():
      module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))


class WanI2V:
  def __init__(self, config, checkpoint_dir, low_noise_loras=(), high_noise_loras=()):
    """
    Args:
      low_noise_loras / high_noise_loras: iterable of (path, strength) tuples to merge
        into the respective DiT before inference.
    """
    self.device = torch.device("cuda:0")
    self.config = config

    self.num_train_timesteps = config.num_train_timesteps
    self.boundary = config.boundary
    self.param_dtype = config.param_dtype
    self.text_len = config.text_len

    self.text_encoder = T5EncoderModel(
      text_len=config.text_len,
      dtype=config.t5_dtype,
      device=torch.device('cpu'),
      checkpoint_path=_resolve_comfyui_file(checkpoint_dir, config.t5_checkpoint, "text_encoders"),
      tokenizer_path=_resolve_t5_tokenizer_path(checkpoint_dir, config.t5_tokenizer),
    )

    self.vae_stride = config.vae_stride
    self.patch_size = config.patch_size
    self.vae = Wan2_1_VAE(
      vae_pth=_resolve_comfyui_file(
        checkpoint_dir, config.vae_checkpoint, "vae", _COMFYUI_VAE_CHECKPOINTS.get(config.vae_checkpoint, ())
      ),
      device=self.device,
    )

    logging.info("Creating WanModel")
    self.low_noise_model = self._load_dit(_LOW_NOISE_I2V_CHECKPOINT, low_noise_loras, config)
    self.high_noise_model = self._load_dit(_HIGH_NOISE_I2V_CHECKPOINT, high_noise_loras, config)

    self.sample_neg_prompt = config.sample_neg_prompt

  def _load_dit(self, checkpoint, loras, config):
    model = _load_i2v_wan_model(checkpoint, config)
    for lora_path, strength in loras:
      _merge_lora_into_wan_model(model, lora_path, strength=strength)
    model.eval().requires_grad_(False)
    return model

  def _prepare_model_for_timestep(self, t, boundary, offload_model):
    if t.item() >= boundary:
      required_name, offload_name = 'high_noise_model', 'low_noise_model'
    else:
      required_name, offload_name = 'low_noise_model', 'high_noise_model'
    required = getattr(self, required_name)
    offload = getattr(self, offload_name)
    if offload_model and next(offload.parameters()).device.type == 'cuda':
      offload.to('cpu')
    if next(required.parameters()).device.type == 'cpu':
      required.to(self.device)
    return required

  def generate(
    self,
    input_prompt,
    img,
    max_area=720 * 1280,
    frame_num=81,
    shift=5.0,
    sample_solver='unipc',
    sampling_steps=40,
    guide_scale=5.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
  ):
    r"""
    Generates video frames from input image and text prompt using diffusion process.

    Args:
        input_prompt (`str`):
            Text prompt for content generation.
        img (PIL.Image.Image):
            Input image tensor. Shape: [3, H, W]
        max_area (`int`, *optional*, defaults to 720*1280):
            Maximum pixel area for latent space calculation. Controls video resolution scaling
        frame_num (`int`, *optional*, defaults to 81):
            How many frames to sample from a video. The number should be 4n+1
        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
            [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
        sample_solver (`str`, *optional*, defaults to 'unipc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 40):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
            Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            If tuple, the first guide_scale will be used for low noise model and
            the second guide_scale will be used for high noise model.
        n_prompt (`str`, *optional*, defaults to ""):
            Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
        seed (`int`, *optional*, defaults to -1):
            Random seed for noise generation. If -1, use random seed
        offload_model (`bool`, *optional*, defaults to True):
            If True, offloads models to CPU during generation to save VRAM

    Returns:
        torch.Tensor:
            Generated video frames tensor. Dimensions: (C, N H, W) where:
            - C: Color channels (3 for RGB)
            - N: Number of frames (81)
            - H: Frame height (from max_area)
            - W: Frame width from max_area)
    """
    # preprocess
    guide_scale = (guide_scale, guide_scale) if isinstance(guide_scale, float) else guide_scale
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]

    max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
      16, (F - 1) // self.vae_stride[0] + 1, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device
    )

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
      n_prompt = self.sample_neg_prompt

    self.text_encoder.model.to(self.device)
    context = self.text_encoder([input_prompt], self.device)
    context_null = self.text_encoder([n_prompt], self.device)
    if offload_model:
      self.text_encoder.model.cpu()

    y = self.vae.encode(
      [
        torch.concat(
          [
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, h, w),
          ],
          dim=1,
        ).to(self.device)
      ]
    )[0]
    y = torch.concat([msk, y])

    with torch.amp.autocast('cuda', dtype=self.param_dtype), torch.no_grad():
      boundary = self.boundary * self.num_train_timesteps

      if sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepScheduler(
          num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps
      elif sample_solver == 'dpm++':
        sample_scheduler = FlowDPMSolverMultistepScheduler(
          num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
      else:
        raise NotImplementedError("Unsupported solver.")

      # sample videos
      latent = noise

      def _pad_context(ctx_list):
        ctx = ctx_list[0]
        if ctx.size(0) < self.text_len:
          ctx = torch.cat([ctx, ctx.new_zeros(self.text_len - ctx.size(0), ctx.size(1))])
        return ctx.unsqueeze(0).to(self.param_dtype)

      encoder_hidden_states_cond = _pad_context(context)
      encoder_hidden_states_uncond = _pad_context(context_null)

      if offload_model:
        torch.cuda.empty_cache()

      for _, t in enumerate(tqdm(timesteps)):
        latent_model_input = torch.cat([latent.to(self.device), y], dim=0).unsqueeze(0).to(self.param_dtype)
        timestep = t.to(self.device).reshape(1).unsqueeze(-1).expand(-1, max_seq_len)

        model = self._prepare_model_for_timestep(t, boundary, offload_model)
        sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

        noise_pred_cond = model(latent_model_input, timestep, encoder_hidden_states_cond).squeeze(0)
        if offload_model:
          torch.cuda.empty_cache()
        noise_pred_uncond = model(latent_model_input, timestep, encoder_hidden_states_uncond).squeeze(0)
        if offload_model:
          torch.cuda.empty_cache()
        noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

        temp_x0 = sample_scheduler.step(
          noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g
        )[0]
        latent = temp_x0.squeeze(0)

        x0 = [latent]
        del latent_model_input, timestep

      if offload_model:
        self.low_noise_model.cpu()
        self.high_noise_model.cpu()
        torch.cuda.empty_cache()

      videos = self.vae.decode(x0)

    del noise, latent, x0
    del sample_scheduler
    if offload_model:
      gc.collect()
      torch.cuda.synchronize()

    return videos[0]
