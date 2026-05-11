import logging
from pathlib import Path

import requests
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from .rife_arch import IFNet

CKPT_CONFIGS = {
  "rife49": {
    "url": "https://huggingface.co/simonri/saferife/resolve/main/rife49.safetensors",
    "file": "rife49.safetensors",
  },
}

SCALE_LIST = (8.0, 4.0, 2.0, 1.0)


class RIFE:
  """RIFE (Real-Time Intermediate Flow Estimation) frame interpolation."""

  def __init__(self, ckpt_name="rife49", device=None, cache_dir=None):
    assert ckpt_name in CKPT_CONFIGS, f"Unknown checkpoint {ckpt_name}, choices: {list(CKPT_CONFIGS)}"

    default_cache = Path(__file__).parent.parent / "cache" / "rife_models"
    self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.ckpt_name = ckpt_name
    self.model = self._load_model(ckpt_name)

  def _download_model(self, ckpt_name):
    config = CKPT_CONFIGS[ckpt_name]
    model_path = self.cache_dir / config["file"]
    if model_path.exists():
      return model_path

    logging.info(f"Downloading RIFE model {ckpt_name} from {config['url']}")
    try:
      response = requests.get(config["url"], stream=True, timeout=120)
      response.raise_for_status()
      total_size = int(response.headers.get("content-length", 0))
      with open(model_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)
          pbar.update(len(chunk))
    except Exception:
      model_path.unlink(missing_ok=True)
      raise
    return model_path

  def _load_model(self, ckpt_name):
    model_path = self._download_model(ckpt_name)
    logging.info(f"Loading RIFE model {ckpt_name} on {self.device}")
    model = IFNet().to(self.device)
    model.load_state_dict(load_file(str(model_path), device=str(self.device)))
    model.eval()
    return model

  @torch.no_grad()
  def interpolate(self, images, multiplier=2, ensemble=True):
    """
    Interpolate frames using RIFE.

    Args:
      images: tensor (N, H, W, C) with values in [0, 1].
      multiplier: number of output frames per input pair (>= 2).
      ensemble: average forward and reversed passes for better quality (slower).

    Returns:
      Tensor (N + (N-1)*(multiplier-1), H, W, C) on the same device as `images`.
    """
    assert multiplier >= 2, f"multiplier must be >= 2, got {multiplier}"
    n = images.shape[0]
    assert n >= 2, f"Need at least 2 frames for interpolation, got {n}"

    logging.info(f"Interpolating {n} frames with {multiplier}x multiplier (ensemble={ensemble})")

    src_device = images.device
    images_dev = images.to(self.device)
    output_frames = []

    for i in tqdm(range(n - 1), desc="RIFE"):
      frame0 = images_dev[i : i + 1]
      frame1 = images_dev[i + 1 : i + 2]
      output_frames.append(frame0)

      img0 = frame0.permute(0, 3, 1, 2)
      img1 = frame1.permute(0, 3, 1, 2)

      for j in range(1, multiplier):
        pred = self.model(img0, img1, timestep=j / multiplier, scale_list=SCALE_LIST, ensemble=ensemble)
        pred = pred.permute(0, 2, 3, 1).clamp(0, 1)
        output_frames.append(pred)

    output_frames.append(images_dev[-1:])
    result = torch.cat(output_frames, dim=0).to(src_device)
    logging.info(f"Interpolation complete: {n} -> {result.shape[0]} frames")
    return result
