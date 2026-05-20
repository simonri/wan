from wan.configs.pipeline.wan import WanI2VConfig
from wan.modules.t5 import T5EncoderModel
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.pipeline.base import PipelineBase
from wan.platform import get_local_torch_device
from wan.stages.image_encoding import ImageVAEEncodingStage
from wan.stages.input_validation import InputValidationStage
from wan.stages.text_encoding import TextEncodingStage


class WanImageToVideoPipeline(PipelineBase):
  def __init__(self, config: WanI2VConfig):
    super().__init__()
    self.config = config

    print("Loading pipeline modules...")
    self.modules = self.load_modules()

    self.create_pipeline_stages(config)

  def load_modules(self, config: WanI2VConfig) -> dict[str, any]:
    local_torch_device = get_local_torch_device()

    text_encoder = T5EncoderModel(
      text_len=config.dit_config.text_len,
      dtype=config.dit_config.t5_dtype,
      checkpoint_path=config.dit_config.t5_checkpoint,
    )

    tokenizer = HuggingfaceTokenizer(name=config.dit_config.t5_tokenizer, seq_len=config.dit_config.text_len)

    vae = Wan2_1_VAE(vae_pth=config.dit_config.vae_checkpoint, device=local_torch_device)

    return {
      "text_encoder": text_encoder,
      "tokenizer": tokenizer,
      "vae": vae,
    }

  def get_module(self, name: str) -> any:
    return self.modules[name]

  def create_pipeline_stages(self, config: WanI2VConfig):
    self.add_stage(InputValidationStage())

    self.add_stage(
      TextEncodingStage(text_encoder=self.get_module("text_encoder"), tokenizer=self.get_module("tokenizer"))
    )

    self.add_stage(ImageVAEEncodingStage(vae=self.get_module("vae")))
