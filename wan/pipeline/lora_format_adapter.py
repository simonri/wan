from collections.abc import Mapping
from enum import Enum

import torch


class LoRAFormat(str, Enum):
  STANDARD = "standard"  # diffusers/PEFT style: <path>.lora_A.weight / <path>.lora_B.weight
  WAN = "wan"  # original Wan: diffusion_model.<path>.lora_down.weight / .lora_up.weight
  KOHYA = "kohya"  # A1111/sd-scripts: lora_unet_<path-with-_>.lora_down.weight / .lora_up.weight


KOHYA_PREFIXES = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")


def detect_lora_format(state_dict: Mapping[str, torch.Tensor]) -> LoRAFormat:
  keys = list(state_dict.keys())
  if not keys:
    return LoRAFormat.STANDARD

  # WAN format is identified by the diffusion_model. prefix regardless of
  # whether the file uses lora_A/lora_B or lora_down/lora_up naming — check
  # this before the lora_A presence test, which would otherwise short-circuit.
  if any(k.startswith("diffusion_model.") for k in keys):
    return LoRAFormat.WAN

  if any(".lora_A." in k or ".lora_B." in k for k in keys):
    return LoRAFormat.STANDARD

  # majority-rule so a stray metadata key doesn't disqualify a file
  if sum(k.startswith(KOHYA_PREFIXES) for k in keys) > len(keys) // 2:
    return LoRAFormat.KOHYA

  return LoRAFormat.STANDARD


def _swap_down_up_to_A_B(name: str) -> str:
  return name.replace("lora_down.weight", "lora_A.weight").replace("lora_up.weight", "lora_B.weight")


def _normalize_kohya(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  out: dict[str, torch.Tensor] = {}
  for name, weight in state_dict.items():
    # text-encoder LoRAs aren't supported here; skip them entirely
    if name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
      continue
    if name.startswith("lora_unet_"):
      name = name[len("lora_unet_") :]
    out[_swap_down_up_to_A_B(name)] = weight
  return out


def _normalize_wan(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  out: dict[str, torch.Tensor] = {}
  for name, weight in state_dict.items():
    name = name.removeprefix("diffusion_model.")
    out[_swap_down_up_to_A_B(name)] = weight
  return out


def normalize_lora_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  fmt = detect_lora_format(state_dict)
  print(f"Detected LoRA format: {fmt}")

  if fmt == LoRAFormat.KOHYA:
    return _normalize_kohya(state_dict)
  if fmt == LoRAFormat.WAN:
    return _normalize_wan(state_dict)
  return dict(state_dict)
