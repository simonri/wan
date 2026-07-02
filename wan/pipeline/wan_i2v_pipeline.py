import threading
import time

import torch
from transformers import AutoTokenizer

from wan.modules.model import WanModel
from wan.modules.t5 import T5Encoder
from wan.modules.wanvae import Wan2_1_VAE
from wan.pipeline.base import PipelineBase
from wan.pipeline.executor import BaseExecutor
from wan.pipeline.lora_pipeline import LoRAPipeline
from wan.platform import CudaPlatform, get_local_torch_device
from wan.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from wan.server_args import ServerArgs
from wan.stages.decoding import DecodingStage
from wan.stages.denoising import DenoisingStage
from wan.stages.image_encoding import ImageVAEEncodingStage
from wan.stages.input_validation import InputValidationStage
from wan.stages.latent_preparation import LatentPreparationStage
from wan.stages.text_encoding import LazyTextEncoder, TextEncodingStage
from wan.stages.timestep_preparation import TimestepPreparationStage
from wan.torch_utils import PRECISION_TO_TYPE, set_default_torch_dtype


def _prefetch_file(path: str, chunk_bytes: int = 1 << 24) -> None:
  """Sequentially read a file to pull it into the page cache (reads release the
  GIL, so this overlaps with the main thread's model load)."""
  try:
    with open(path, "rb", buffering=0) as f:
      while f.read(chunk_bytes):
        pass
  except OSError as e:
    print(f"  prefetch {path} failed: {e}")


class WanImageToVideoPipeline(LoRAPipeline, PipelineBase):
  def __init__(self, server_args: ServerArgs, executor: BaseExecutor):
    super().__init__(executor)
    print("Loading pipeline modules...")
    self.modules = self.load_modules(server_args)

    self.create_pipeline_stages(server_args)

  def load_modules(self, server_args: ServerArgs) -> dict[str, any]:
    pipeline_config = server_args.pipeline_config
    local_torch_device = get_local_torch_device()

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    print(f"  tokenizer init: {time.perf_counter() - t0:.2f}s")

    # T5 and VAE: meta + assign=True + direct-to-GPU read (fast for small/medium files)
    text_encoder_dtype = PRECISION_TO_TYPE[pipeline_config.text_encoder_precision]

    def build_text_encoder() -> T5Encoder:
      t0 = time.perf_counter()
      with torch.device("meta"), set_default_torch_dtype(text_encoder_dtype):
        encoder = T5Encoder(config=pipeline_config.text_encoder_config)
      print(f"  T5 construct (meta): {time.perf_counter() - t0:.2f}s")
      encoder.load("models/text_encoders/umt5-xxl-enc-bf16.safetensors", server_args)
      return encoder

    text_encoder = LazyTextEncoder(build_text_encoder)

    vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]
    t0 = time.perf_counter()
    with torch.device("meta"), set_default_torch_dtype(vae_dtype):
      vae = Wan2_1_VAE(config=pipeline_config.vae_config)
    print(f"  VAE construct (meta): {time.perf_counter() - t0:.2f}s")
    vae.load("models/vae/wan_2.1_vae.safetensors", server_args)

    scheduler = FlowMatchEulerDiscreteScheduler(
      shift=pipeline_config.flow_shift,
    )

    # Transformers: pre-allocate on GPU and copy CPU state_dict in.
    # Direct-to-GPU safetensors and post-load .to(device) were both significantly slower
    # for 28GB checkpoints; the CPU-staged + in-place copy_ path is fastest empirically.
    transformer_dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]

    # Warm the page cache for transformer_2's file while transformer 1 loads —
    # the disk (~300 MB/s virtio) is the bottleneck, so overlapping the two
    # sequential 28 GB reads nearly halves cold-start load time.
    low_noise_path = "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.flashpack"
    prefetch_thread = threading.Thread(target=_prefetch_file, args=(low_noise_path,), daemon=True)
    prefetch_thread.start()

    t0 = time.perf_counter()
    print(f"  Transformer construct (meta): avail mem before: {CudaPlatform.get_available_gpu_memory():.2f} GB")
    with torch.device("meta"), set_default_torch_dtype(transformer_dtype):
      transformer = WanModel(config=pipeline_config.dit_config)
    print(f"  Transformer construct (meta): {time.perf_counter() - t0:.2f}s")
    transformer.load(
      "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.flashpack", server_args, device=local_torch_device
    )

    t0 = time.perf_counter()
    print(f"  Transformer_2 construct (meta): avail mem before: {CudaPlatform.get_available_gpu_memory():.2f} GB")
    with torch.device("meta"), set_default_torch_dtype(transformer_dtype):
      transformer_2 = WanModel(config=pipeline_config.dit_config)
    print(f"  Transformer_2 construct (meta): {time.perf_counter() - t0:.2f}s")
    prefetch_thread.join(timeout=600)
    transformer_2.load(low_noise_path, server_args, device=local_torch_device)

    print(f"== total load_modules: {time.perf_counter() - t_total:.2f}s ==")

    return {
      "text_encoder": text_encoder,
      "tokenizer": tokenizer,
      "vae": vae,
      "scheduler": scheduler,
      "transformer": transformer,
      "transformer_2": transformer_2,
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

    self.add_stage(
      DenoisingStage(
        transformer=self.get_module("transformer"),
        transformer_2=self.get_module("transformer_2"),
        scheduler=self.get_module("scheduler"),
      )
    )

    self.add_stage(DecodingStage(vae=self.get_module("vae")))
