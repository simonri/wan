import dataclasses
import hashlib
import json
import os
import pathlib

import torch


def _prompt_key(prompt: str) -> str:
  return hashlib.sha256(prompt.encode()).hexdigest()


def _namespace_path(cache_dir: str, namespace: str) -> pathlib.Path:
  return pathlib.Path(cache_dir) / namespace


class TextEmbedCache:
  def build_namespace(self, pipeline_config) -> str:
    arch = pipeline_config.text_encoder_config.arch_config
    fields = dataclasses.asdict(arch) if dataclasses.is_dataclass(arch) else vars(arch)
    # Exclude non-serializable or mutable fields
    serializable = {k: v for k, v in fields.items() if isinstance(v, (int, float, bool, str))}
    digest = hashlib.sha256(json.dumps(serializable, sort_keys=True).encode()).hexdigest()[:16]
    return digest

  def load(
    self,
    cache_dir: str | None,
    namespace: str,
    prompt: str,
    target_device: torch.device | str,
    expected_shape: tuple[int, ...],
  ) -> torch.Tensor | None:
    if cache_dir is None:
      return None
    path = _namespace_path(cache_dir, namespace) / f"{_prompt_key(prompt)}.pt"
    if not path.exists():
      return None
    tensor = torch.load(path, map_location=target_device, weights_only=True)
    if tensor.shape != torch.Size(expected_shape):
      return None
    return tensor

  def save(
    self,
    cache_dir: str | None,
    namespace: str,
    prompt: str,
    embedding: torch.Tensor,
  ) -> None:
    if cache_dir is None:
      return
    dir_path = _namespace_path(cache_dir, namespace)
    os.makedirs(dir_path, exist_ok=True)
    path = dir_path / f"{_prompt_key(prompt)}.pt"
    torch.save(embedding.cpu(), path)


text_embed_cache = TextEmbedCache()
