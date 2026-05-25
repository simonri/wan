from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer

from wan.cache import text_embed_cache
from wan.configs.pipeline.base import BaseEncoderOutput
from wan.modules.t5 import T5Encoder
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


@dataclass(frozen=True)
class TextEncodingFingerprint:
  prompt: Any
  prompt_template: Any


class LazyTextEncoder:
  """Defers T5 construction and weight load until the first cache miss."""

  def __init__(self, builder: Callable[[], T5Encoder]):
    self._builder = builder
    self._encoder: T5Encoder | None = None

  def get(self) -> T5Encoder:
    if self._encoder is None:
      self._encoder = self._builder()
    return self._encoder


class TextEncodingStage(PipelineStage):
  deduplicated_output_fields = (
    "prompt_embeds",
    "prompt_attention_mask",
    "prompt_embeds_mask",
    "prompt_seq_lens",
    "pooled_embeds",
  )

  def __init__(self, text_encoder: LazyTextEncoder, tokenizer: AutoTokenizer):
    super().__init__()
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder

  @torch.no_grad()
  def forward(self, batch: Req, server_args: ServerArgs):
    batch.prompt_embeds = self.encode_text(batch.prompt, server_args)

    return batch

  @torch.no_grad()
  def encode_text(
    self, text: str | list[str], server_args: ServerArgs, device: torch.device | str | None = None
  ) -> torch.Tensor:
    """Encode prompts with T5/UMT5, returning a [B, text_len, D] tensor.

    Per-prompt embeddings are cached on disk under server_args.text_embed_cache_dir.
    On full cache hit the T5 weights are never loaded.
    """
    target_device = device if device is not None else get_local_torch_device()

    if isinstance(text, str):
      text = [text]

    pipeline_config = server_args.pipeline_config
    arch = pipeline_config.text_encoder_config.arch_config
    expected_shape = (arch.text_len, arch.dim)
    cache_dir = server_args.text_embed_cache_dir
    namespace = text_embed_cache.build_namespace(pipeline_config)

    cached: list[torch.Tensor | None] = [
      text_embed_cache.load(cache_dir, namespace, p, target_device, expected_shape) for p in text
    ]
    miss_indices = [i for i, t in enumerate(cached) if t is None]

    if miss_indices:
      miss_texts = [text[i] for i in miss_indices]
      encoded = self._encode_uncached(miss_texts, server_args, target_device)
      for k, i in enumerate(miss_indices):
        emb = encoded[k]
        cached[i] = emb
        text_embed_cache.save(cache_dir, namespace, text[i], emb)

    return torch.stack(cached, dim=0)

  @torch.no_grad()
  def _encode_uncached(
    self, text: list[str], server_args: ServerArgs, target_device: torch.device | str
  ) -> torch.Tensor:
    pipeline_config = server_args.pipeline_config
    tok_kwargs = pipeline_config.text_encoder_config.arch_config.tokenizer_kwargs

    text_inputs = pipeline_config.tokenize_prompt(text, self.tokenizer, tok_kwargs).to(target_device)
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs.get("attention_mask")

    encoder = self.text_encoder.get()
    last_hidden_state = encoder(input_ids, attention_mask)  # [B, L, D]
    outputs = BaseEncoderOutput(last_hidden_state=last_hidden_state, attention_masks=attention_mask)
    return pipeline_config.postprocess_text(outputs, text_inputs)

  def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs) -> TextEncodingFingerprint:
    return TextEncodingFingerprint(
      prompt=self.freeze_for_dedup(batch.prompt),
      prompt_template=self.freeze_for_dedup(batch.prompt_template),
    )
