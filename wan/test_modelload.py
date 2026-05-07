import safetensors
import torch
from safetensors.torch import load_file as safetensors_load_file

from .modules.vae2_1 import WanVAE_

pretrained_path = "models/vae/wan_2.1_vae.safetensors"
device = torch.device("cuda:0")

# test loading
cfg = dict(dim=96, z_dim=None, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)

# init model
with torch.device('meta'):
  model = WanVAE_(**cfg)


def load_torch_file(ckpt):
  result = {}
  with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
    sd = {}
    for k in f.offset_keys():
      sd[k] = f.get_tensor(k)
  return result

# load checkpoint
loaded = {}
loaded.update(safetensors_load_file(pretrained_path))


# assign=True ?
model.load_state_dict(loaded, strict=False)
