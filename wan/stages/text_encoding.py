import torch
from transformers import AutoTokenizer

from wan.configs.pipeline.base import BaseEncoderOutput
from wan.modules.t5 import T5Encoder
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.base import PipelineStage
from wan.stages.schedule_batch import Req


class TextEncodingStage(PipelineStage):
  def __init__(self, text_encoder: T5Encoder, tokenizer: AutoTokenizer):
    super().__init__()
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder

  @torch.no_grad()
  def forward(self, batch: Req, server_args: ServerArgs):
    batch.prompt_embeds = self.encode_text(batch.prompt, server_args)

    if batch.do_classifier_free_guidance:
      raise NotImplementedError("Classifier-free guidance is not implemented yet")

    return batch

  @torch.no_grad()
  def encode_text(
    self, text: str | list[str], server_args: ServerArgs, device: torch.device | str | None = None
  ) -> torch.Tensor:
    """Encode prompts with T5/UMT5, returning a [B, text_len, D] tensor."""
    target_device = device if device is not None else get_local_torch_device()

    if isinstance(text, str):
      text = [text]

    pipeline_config = server_args.pipeline_config
    tok_kwargs = pipeline_config.text_encoder_config.arch_config.tokenizer_kwargs

    text_inputs = pipeline_config.tokenize_prompt(text, self.tokenizer, tok_kwargs).to(target_device)
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs.get("attention_mask")

    last_hidden_state = self.text_encoder(input_ids, attention_mask)  # [B, L, D]
    outputs = BaseEncoderOutput(last_hidden_state=last_hidden_state, attention_masks=attention_mask)
    return pipeline_config.postprocess_text(outputs, text_inputs)
