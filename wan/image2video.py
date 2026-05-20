# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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

from wan.configs.pipeline.wan import WanI2VConfig
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.stages.schedule_batch import Req
from wan.stages.text_encoding import TextEncodingStage
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

_LOW_NOISE_I2V_CHECKPOINT = "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
_HIGH_NOISE_I2V_CHECKPOINT = "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"


def _linear_to_shifted_sigma(sigma_linear, shift):
  """Flow-matching reparameterization: σ_shifted = shift·σ / (1 + (shift-1)·σ)."""
  return shift * sigma_linear / (1 + (shift - 1) * sigma_linear)


def _create_i2v_wan_model(arch):
  with init_empty_weights():
    return WanModel(
      patch_size=arch.patch_size,
      in_dim=arch.in_dim,
      dim=arch.hidden_size,
      ffn_dim=arch.ffn_dim,
      freq_dim=arch.freq_dim,
      out_dim=arch.num_channels_latents,
      num_heads=arch.num_attention_heads,
      num_layers=arch.num_layers,
      qk_norm=arch.qk_norm,
      eps=arch.eps,
    )


def _load_i2v_wan_model(checkpoint_path, arch):
  if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Could not find I2V diffusion checkpoint: {checkpoint_path}")

  logging.info(f"Loading WanModel from safetensors checkpoint: {checkpoint_path}")
  model = _create_i2v_wan_model(arch)
  state_dict = load_safetensors_file(checkpoint_path, device="cpu")
  for k in list(state_dict.keys()):
    state_dict[k] = state_dict.pop(k).to(arch.param_dtype)

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
  for i in range(arch.num_layers):
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


def _load_lora_state_dict(lora_path):
  """Load and validate a LoRA safetensors file into a CPU state_dict."""
  if not os.path.isfile(lora_path):
    raise FileNotFoundError(f"Could not find WanModel LoRA checkpoint: {lora_path}")
  return load_safetensors_file(lora_path, device="cpu")


def _apply_lora_state_dict(model, state_dict, strength=1.0):
  """Merge a pre-loaded LoRA state_dict into model weights in-place.

  Linear in `strength`: calling with `-strength` reverses a prior `+strength` merge
  (modulo fp roundoff). Used to toggle speedup LoRAs across stage boundaries.
  """
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


def _merge_lora_into_wan_model(model, lora_path, strength=1.0):
  """Convenience wrapper: load from disk and apply in one call."""
  logging.info(f"Merging WanModel LoRA from safetensors checkpoint: {lora_path}")
  _apply_lora_state_dict(model, _load_lora_state_dict(lora_path), strength=strength)


class WanI2V:
  def __init__(self, config: WanI2VConfig, low_noise_loras=(), high_noise_loras=()):
    """
    Args:
      config: WanI2VConfig (pipeline-level, wraps the DiT + sampler config).
      low_noise_loras / high_noise_loras: iterable of (path, strength) tuples merged
        into the respective DiT at init.
    """
    self.config = config
    dit_cfg = config.dit_config
    arch = dit_cfg.arch_config
    self.device = torch.device("cuda:0")

    self.text_encoder = T5EncoderModel(
      text_len=dit_cfg.text_len,
      dtype=dit_cfg.t5_dtype,
      checkpoint_path=dit_cfg.t5_checkpoint,
    )

    self.tokenizer = HuggingfaceTokenizer(name=dit_cfg.t5_tokenizer, seq_len=dit_cfg.text_len, clean='whitespace')

    self.vae = Wan2_1_VAE(vae_pth=dit_cfg.vae_checkpoint, device=self.device)

    logging.info("Creating WanModel")
    self.low_noise_model = self._load_dit(_LOW_NOISE_I2V_CHECKPOINT, low_noise_loras, arch)
    self.high_noise_model = self._load_dit(_HIGH_NOISE_I2V_CHECKPOINT, high_noise_loras, arch)

    # stages
    self.text_encoding_stage = TextEncodingStage(text_encoder=self.text_encoder, tokenizer=self.tokenizer)

  def _load_dit(self, checkpoint, loras, arch):
    model = _load_i2v_wan_model(checkpoint, arch)
    for lora_path, strength in loras:
      _merge_lora_into_wan_model(model, lora_path, strength=strength)
    model.eval().requires_grad_(False)
    return model

  def _stage_for_timestep(self, t, boundary):
    """Returns 'high' for timesteps at or above the boundary, otherwise 'low'."""
    return 'high' if t.item() >= boundary else 'low'

  def _prepare_model_for_timestep(self, t, boundary):
    stage = self._stage_for_timestep(t, boundary)
    required = self.low_noise_model if stage == 'low' else self.high_noise_model
    if next(required.parameters()).device.type == 'cpu':
      required.to(self.device)
    return required, stage

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
    boundary=None,
    n_prompt="",
    seed=-1,
  ):
    r"""
    Generates video frames from input image and text prompt using diffusion process.

    Args:
      input_prompt: Text prompt for content generation.
      img (PIL.Image.Image): Input image. Shape: [3, H, W].
      max_area: Maximum pixel area for the generated latent. Controls output resolution.
      frame_num: How many frames to sample. Must be 4n+1.
      shift: Flow-matching schedule shift parameter.
      sample_solver: 'unipc' or 'dpm++'.
      sampling_steps: Number of diffusion sampling steps.
      guide_scale: Classifier-free guidance scale. A float (applied to both stages) or
          a 2-tuple (low_cfg, high_cfg).
      boundary: Linear-sigma boundary between low- and high-noise stages.
          Shift-invariant; the shifted-sigma timestep is derived internally. Overrides config.
      n_prompt: Negative prompt. Defaults to `config.sample_neg_prompt`.
      seed: -1 picks a random seed.

    Returns:
      torch.Tensor of shape (C, N, H, W).
    """
    # preprocess
    if isinstance(guide_scale, (int, float)):
      guide_scale_low = guide_scale_high = float(guide_scale)
    else:
      guide_scale = tuple(float(g) for g in guide_scale)
      if len(guide_scale) == 1:
        guide_scale_low = guide_scale_high = guide_scale[0]
      elif len(guide_scale) == 2:
        guide_scale_low, guide_scale_high = guide_scale
      else:
        raise ValueError(f"guide_scale must have 1 or 2 values, got {len(guide_scale)}")
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

    num_frames = frame_num
    vae_stride = self.config.dit_config.vae_stride
    patch_size = self.config.dit_config.arch_config.patch_size
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]

    num_lat_frames = (num_frames - 1) // vae_stride[0] + 1

    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(16, num_lat_frames, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)

    msk = torch.ones(1, num_frames, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if n_prompt == "":
      n_prompt = self.config.dit_config.sample_neg_prompt

    self.text_encoder.model.to(self.device)
    context = self.text_encoder([input_prompt], self.device)
    context_null = self.text_encoder([n_prompt], self.device)

    req = Req()

    req.prompt = input_prompt
    req.do_classifier_free_guidance = False

    context = self.text_encoding_stage(req)

    with torch.no_grad():
      # Bicubic + zeros allocated on CPU then transferred as a single tensor: keeps the
      # pre-encode conditioning out of GPU memory at 720x1280, F=81 (~1 GB) where it
      # would otherwise push past the model's working-set headroom.
      conditioning_input = torch.concat(
        [
          torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
          torch.zeros(3, num_frames - 1, h, w),
        ],
        dim=1,
      ).to(self.device)
      y = self.vae.encode([conditioning_input])[0]
      y = torch.concat([msk, y])

    dit_cfg = self.config.dit_config
    param_dtype = dit_cfg.arch_config.param_dtype
    num_train_timesteps = dit_cfg.num_train_timesteps
    text_len = dit_cfg.text_len

    with torch.amp.autocast('cuda', dtype=param_dtype), torch.no_grad():
      boundary_lin = self.config.boundary_ratio if boundary is None else boundary
      boundary_t = _linear_to_shifted_sigma(boundary_lin, shift) * num_train_timesteps

      if sample_solver == 'unipc':
        sample_scheduler = FlowUniPCMultistepScheduler(
          num_train_timesteps=num_train_timesteps, shift=self.config.flow_shift
        )
        sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
        timesteps = sample_scheduler.timesteps
      elif sample_solver == 'dpm++':
        sample_scheduler = FlowDPMSolverMultistepScheduler(
          num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device, sigmas=sampling_sigmas)
      else:
        raise NotImplementedError("Unsupported solver.")

      # sample videos
      latent = noise

      def _pad_context(ctx_list):
        ctx = ctx_list[0]
        if ctx.size(0) < text_len:
          ctx = torch.cat([ctx, ctx.new_zeros(text_len - ctx.size(0), ctx.size(1))])
        return ctx.unsqueeze(0).to(param_dtype)

      encoder_hidden_states_cond = _pad_context(context)
      encoder_hidden_states_uncond = _pad_context(context_null)

      stage_to_cfg = {'high': guide_scale_high, 'low': guide_scale_low}
      for t in tqdm(timesteps):
        latent_model_input = torch.cat([latent.to(self.device), y], dim=0).unsqueeze(0).to(param_dtype)
        # 1-D timestep keeps the model on its non-flex modulation path: the 6 shift/scale
        # tensors per block stay [B, 1, dim] and broadcast in the norm kernel, instead of
        # materializing as [B, seq_len, dim] (~3.7 GB per block at 720x1280, F=81).
        timestep = t.to(self.device).reshape(1)

        model, stage = self._prepare_model_for_timestep(t, boundary_t)
        sample_guide_scale = stage_to_cfg[stage]

        noise_pred_cond = model(latent_model_input, timestep, encoder_hidden_states_cond).squeeze(0)
        if sample_guide_scale == 1.0:
          noise_pred = noise_pred_cond
        else:
          noise_pred_uncond = model(latent_model_input, timestep, encoder_hidden_states_uncond).squeeze(0)
          noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

        latent_next = sample_scheduler.step(
          noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=seed_g
        )[0]
        latent = latent_next.squeeze(0)
        del latent_model_input, timestep

      x0 = [latent]

      videos = self.vae.decode(x0)

    del noise, latent, x0
    del sample_scheduler

    return videos[0]
