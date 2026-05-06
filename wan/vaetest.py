import torch
from PIL import Image

from .modules.vae2_1 import Wan2_1_VAE

vae_pth = "models/vae/wan_2.1_vae.safetensors"
device = torch.device("cuda:0")

# load vae
vae = Wan2_1_VAE(
  vae_pth=vae_pth,
  device=device
)

img_path = "examples/i2v_input.JPG"
img = Image.open(img_path).convert("RGB")
img = img.resize((480, 832))

import torchvision.transforms.functional as TF

img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
h, w = img.shape[1:]

print(f"Image size: {h}x{w}")

frame_num = 81

def run_once():
  # encode
  y = vae.encode(
    [
      torch.concat([torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1), torch.zeros(3, frame_num - 1, h, w)], dim=1).to(device)
    ]
  )[0]
  print(y.shape)

  return y



# warmup
for _ in range(2):
  _ = run_once()

import time

# benchmark
torch.cuda.synchronize()
start = time.perf_counter()

run_count = 10
for _ in range(run_count):
  _ = run_once()

torch.cuda.synchronize()
end = time.perf_counter()

print(f"Avg time per forward: {(end - start) / run_count:.6f} sec")
