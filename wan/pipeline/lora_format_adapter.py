from collections.abc import Iterable, Mapping
from enum import StrEnum

import torch
from diffusers.loaders import lora_conversion_utils as lcu


class LoRAFormat(StrEnum):
  STANDARD = "standard"
  NON_DIFFUSERS_SD = "non-diffusers-sd"


def _has_substring_key(keys: Iterable[str], substring: str) -> bool:
  return any(substring in k for k in keys)


def _sample_keys(keys: Iterable[str], k: int = 20) -> list[str]:
  out = []
  for i, key in enumerate(keys):
    if i >= k:
      break
    out.append(key)
  return out


def _looks_like_non_diffusers_sd(state_dict: Mapping[str, torch.Tensor]) -> bool:
  """
  Classic non-diffusers SD LoRA (Kohya/A1111)
  """
  if not state_dict:
    return False
  keys = state_dict.keys()
  return all(k.startswith(("lora_unet_", "lora_te_", "lora_te1_ ", "lora_te2_")) for k in keys)


def detect_lora_format_from_state_dict(
  state_dict: Mapping[str, torch.Tensor],
) -> LoRAFormat:
  keys = list(state_dict.keys())
  if not keys:
    return LoRAFormat.STANDARD

  if _has_substring_key(keys, ".lora_A") or _has_substring_key(keys, ".lora_B"):
    return LoRAFormat.STANDARD

  if _looks_like_non_diffusers_sd(state_dict):
    return LoRAFormat.NON_DIFFUSERS_SD


def _convert_with_diffusers_utils_if_available(
  state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
  """Use diffusers.lora_conversion_utils if available."""
  try:
    if hasattr(lcu, "maybe_convert_state_dict"):
      converted = lcu.maybe_convert_state_dict(  # type: ignore[attr-defined]
        state_dict
      )
    else:
      converted = dict(state_dict)

    if not isinstance(converted, dict):
      converted = dict(converted)

    sample = _sample_keys(converted.keys(), 20)
    print(
      f"diffusers.lora_conversion_utils converted keys, sample keys (<=20): {', '.join(sample)}",
    )
    return converted
  except Exception as exc:  # pragma: no cover
    print(
      f"diffusers lora_conversion_utils failed, falling back to internal converters. Error: {exc}",
      exc,
    )
    return None


def _convert_non_diffusers_sd_simple(
  state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Generic down/up -> A/B conversion for non-diffusers SD-like formats."""
  out: dict[str, torch.Tensor] = {}

  for name, tensor in state_dict.items():
    new_name = name

    if "lora_down.weight" in new_name:
      new_name = new_name.replace("lora_down.weight", "lora_A.weight")
    elif "lora_up.weight" in new_name:
      new_name = new_name.replace("lora_up.weight", "lora_B.weight")
    elif new_name.endswith(".lora_down"):
      new_name = new_name.replace(".lora_down", ".lora_A")
    elif new_name.endswith(".lora_up"):
      new_name = new_name.replace(".lora_up", ".lora_B")

    out[new_name] = tensor

  sample = _sample_keys(out.keys(), 20)
  print(
    f"after NON_DIFFUSERS_SD simple conversion, sample keys (<=20): {', '.join(sample)}",
  )
  return out


def convert_lora_state_dict_by_format(
  state_dict: Mapping[str, torch.Tensor], fmt: LoRAFormat
) -> dict[str, torch.Tensor]:
  if fmt == LoRAFormat.STANDARD:
    maybe = _convert_with_diffusers_utils_if_available(state_dict)
    if maybe is None:
      maybe = dict(state_dict)
    return maybe

  if fmt == LoRAFormat.NON_DIFFUSERS_SD:
    maybe = _convert_with_diffusers_utils_if_available(state_dict)
    if maybe is None:
      maybe = dict(state_dict)
    return _convert_non_diffusers_sd_simple(maybe)

  print(f"Format {fmt} not supported, returning as is")
  return dict(state_dict)


def normalize_lora_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  keys = list(state_dict.keys())
  print(f"normalize_lora_state_dict called #keys={len(keys)}")

  fmt = detect_lora_format_from_state_dict(state_dict)
  print(f"Detected format: {fmt}")

  normalized = convert_lora_state_dict_by_format(state_dict, fmt)

  norm_keys = list(normalized.keys())
  if norm_keys:
    print(f"After convert, sample keys (<=20): {', '.join(_sample_keys(norm_keys, 20))}")

  return normalized
