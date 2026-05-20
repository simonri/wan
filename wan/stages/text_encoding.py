import torch
from transformers import AutoTokenizer

from wan.modules.t5 import T5EncoderModel
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


class TextEncodingStage(PipelineStage):
  def __init__(self, text_encoder: T5EncoderModel, tokenizer: AutoTokenizer):
    super().__init__()
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder

  @torch.no_grad()
  def forward(self, batch: Req, server_args: ServerArgs):
    prompt_text = batch.prompt

    (prompt_embeds_list, prompt_mask_list, pooled_embeds_list, prompt_embeds_mask_list, prompt_seq_lens_list) = (
      self.encode_text(prompt_text, server_args)
    )

    for pe in prompt_embeds_list:
      batch.prompt_embeds.append(pe)

    for pe in pooled_embeds_list:
      batch.pooled_embeds.append(pe)

    if batch.prompt_attention_mask is None:
      batch.prompt_attention_mask = []
      for am in prompt_mask_list:
        batch.prompt_attention_mask.append(am)

    batch.prompt_embeds_mask = []
    batch.prompt_seq_lens = []
    for mask in prompt_mask_list:
      batch.prompt_embeds_mask.append(mask)
    for seq_lens in prompt_seq_lens_list:
      batch.prompt_seq_lens.append(seq_lens)

    # encode neg prompt only if cfg is enabled
    if batch.do_classifier_free_guidance:
      raise NotImplementedError("Classifier-free guidance is not implemented yet")

    return batch

  @torch.no_grad()
  def encode_text(self, text: str | list[str], server_args: ServerArgs, device: torch.device | str | None = None):
    """Encode prompts with T5/UMT5.

    Returns:
      (embeds_list, attn_masks_list, pooler_embeds_list, embeds_masks_list, seq_lens_list)

      Each list has one entry per input prompt. T5 has no pooler output, so
      `pooler_embeds_list` is filled with None placeholders. For this encoder
      `embeds_masks_list` mirrors `attn_masks_list` (no separate embed mask).
      All tensors are trimmed to the prompt's true length (padding stripped).
    """
    target_device = device if device is not None else get_local_torch_device()

    if isinstance(text, str):
      text = [text]

    encoder_config = server_args.pipeline_config.text_encoder_config
    tok_kwargs = encoder_config.arch_config.tokenizer_kwargs

    text_inputs = server_args.pipeline_config.tokenize_prompt(text, self.tokenizer, tok_kwargs).to(target_device)

    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs.get("attention_mask")

    seq_lens = attention_mask.gt(0).sum(dim=1).long()  # [B]
    context = self.text_encoder.model(input_ids, attention_mask)  # [B, L, D]

    embeds_list = [u[:v] for u, v in zip(context, seq_lens, strict=True)]
    attn_masks_list = [m[:v] for m, v in zip(attention_mask, seq_lens, strict=True)]
    pooled_embeds_list = [None] * len(embeds_list)
    embeds_masks_list = attn_masks_list
    seq_lens_list = seq_lens.tolist()

    return (embeds_list, attn_masks_list, pooled_embeds_list, embeds_masks_list, seq_lens_list)
