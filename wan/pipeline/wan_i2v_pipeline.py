from transformers import AutoTokenizer

from wan.modules.t5 import T5Encoder
from wan.modules.wanvae import Wan2_1_VAE
from wan.pipeline.base import PipelineBase
from wan.pipeline.executor import BaseExecutor
from wan.platform import get_local_torch_device
from wan.server_args import ServerArgs
from wan.stages.image_encoding import ImageVAEEncodingStage
from wan.stages.input_validation import InputValidationStage
from wan.stages.latent_preparation import LatentPreparationStage
from wan.stages.text_encoding import TextEncodingStage
from wan.stages.timestep_preparation import TimestepPreparationStage
from wan.torch_utils import PRECISION_TO_TYPE, set_default_torch_dtype, skip_init_modules
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanImageToVideoPipeline(PipelineBase):
  def __init__(self, server_args: ServerArgs, executor: BaseExecutor):
    super().__init__(executor)
    print("Loading pipeline modules...")
    self.modules = self.load_modules(server_args)

    self.create_pipeline_stages(server_args)

  def load_modules(self, server_args: ServerArgs) -> dict[str, any]:
    pipeline_config = server_args.pipeline_config
    local_torch_device = get_local_torch_device()

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    # init text encoder
    text_encoder_dtype = PRECISION_TO_TYPE[pipeline_config.text_encoder_precision]
    with set_default_torch_dtype(text_encoder_dtype), skip_init_modules():
      text_encoder = T5Encoder(config=pipeline_config.text_encoder_config).to(local_torch_device)
    text_encoder.load("models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth", server_args)

    # init vae
    vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
    with set_default_torch_dtype(vae_dtype), skip_init_modules():
      vae = Wan2_1_VAE(config=pipeline_config.vae_config).to(local_torch_device)
    vae.load("models/vae/wan_2.1_vae.safetensors", server_args)

    scheduler = FlowUniPCMultistepScheduler(
      shift=pipeline_config.flow_shift,
    )

    return {
      "text_encoder": text_encoder,
      "tokenizer": tokenizer,
      "vae": vae,
      "scheduler": scheduler,
    }

  def get_module(self, name: str) -> any:
    return self.modules[name]

  def create_pipeline_stages(self, server_args: ServerArgs):
    self.add_stage(InputValidationStage())

    self.add_stage(
      TextEncodingStage(text_encoder=self.get_module("text_encoder"), tokenizer=self.get_module("tokenizer"))
    )

    self.add_stage(ImageVAEEncodingStage(vae=self.get_module("vae")))

    self.add_stage(LatentPreparationStage(scheduler=self.get_module("scheduler")))

    self.add_stage(TimestepPreparationStage(scheduler=self.get_module("scheduler")))
