import argparse
import os

import flashpack

from wan.configs.pipeline.wan import WanI2VConfig
from wan.modules.model import WanModel
from wan.server_args import ServerArgs
from wan.torch_utils import PRECISION_TO_TYPE, set_default_torch_dtype, skip_init_modules

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to .safetensors file")
args = parser.parse_args()

input_path = args.input
output_path = os.path.splitext(input_path)[0] + ".flashpack"

pipeline_config = WanI2VConfig()
server_args = ServerArgs(pipeline_config=pipeline_config)

transformer_dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]

with set_default_torch_dtype(transformer_dtype), skip_init_modules():
  transformer = WanModel(config=pipeline_config.dit_config)

transformer.load(input_path, server_args)

print(f"Packing to {output_path}")
flashpack.pack_to_file(transformer, output_path, target_dtype=None)
