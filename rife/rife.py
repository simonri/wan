import logging
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm

from rife.rife_arch import IFNet

CKPT_CONFIGS = {
  "flownet": {
    "url": "https://huggingface.co/simonri/saferife/resolve/main/flownet.safetensors",
    "file": "flownet.safetensors",
  },
}

# model_path -> model instance
_MODEL_CACHE: dict[str, "Model"] = {}


def download_model(ckpt_name: str, cache_dir: Path) -> Path:
  config = CKPT_CONFIGS[ckpt_name]
  model_path = cache_dir / config["file"]
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


class Model:
  def __init__(self):
    self.flownet = IFNet()
    self.device_type: str = "cpu"

  def eval(self) -> "Model":
    self.flownet.eval()
    return self

  def device(self) -> torch.device:
    return next(self.flownet.parameters()).device

  def load_model(self, path: str, strip_module_preifx: bool = True) -> None:
    model_path = Path(path)
    state = load_file(str(model_path), device="cpu")
    self.flownet.load_state_dict(state, strict=False)

  def inference(
    self,
    img0: torch.Tensor,
    img1: torch.Tensor,
    scale: float = 1.0,
    timestep: float = 0.5,
  ) -> torch.Tensor:
    """interpolate a single frame bwtween img0 and img1"""
    n, c, h, w = img0.shape

    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    pad = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, pad)
    img1 = F.pad(img1, pad)

    imgs = torch.cat((img0, img1), 1)
    scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]

    with torch.no_grad():
      flow_list, mask, merged = self.flownet(imgs, timestep=timestep, scale_list=scale_list)

    return merged[3][:, :, :h, :w]


class RIFE:
  """RIFE (Real-Time Intermediate Flow Estimation) frame interpolation."""

  def __init__(self, ckpt_name="flownet", device=None, cache_dir=None):
    assert ckpt_name in CKPT_CONFIGS, f"Unknown checkpoint {ckpt_name}, choices: {list(CKPT_CONFIGS)}"

    default_cache = Path(__file__).parent.parent / "cache" / "rife_models"
    self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.ckpt_name = ckpt_name

  def _ensure_model_loaded(self) -> Model:
    model_path = download_model(self.ckpt_name, self.cache_dir)
    self._resolved_path = model_path

    if model_path in _MODEL_CACHE:
      return _MODEL_CACHE[model_path]

    device = torch.device(self.device)
    model = Model()
    model.load_model(model_path)
    model.eval()
    model.flownet = model.flownet.to(device)
    return model

  @staticmethod
  def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)

  @staticmethod
  def _tensor_to_frame(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)

  def _make_inference(
    self, model: Model, I0: torch.Tensor, I1: torch.Tensor, n: int, scale: float
  ) -> list[torch.Tensor]:
    if n == 1:
      return [model.inference(I0, I1, scale=scale)]
    mid = model.inference(I0, I1, scale=scale)
    return (
      self._make_inference(model, I0, mid, n // 2, scale) + [mid] + self._make_inference(model, mid, I1, n // 2, scale),
    )

  def interpolate(
    self,
    frames: list[np.ndarray],
    exp: int = 1,
    scale: float = 1.0,
  ) -> tuple[list[np.ndarray], int]:
    """
    Interpolate frames using RIFE.

    Args:
      frames: List of uint8 numpy arrays with shape [H, W, 3].
      exp:    Exponent for interpolation factor. 1 → 2×, 2 → 4×.
      scale:  RIFE inference scale. Use 0.5 for high-resolution inputs.

    Returns:
      (interpolated_frames, multiplier) where multiplier = 2**exp.
    """
    if len(frames) < 2:
      logging.warning("Frame interpolation requires at least 2 frames; returning input unchanged.")
      return frames, 1

    model = self._ensure_model_loaded()
    device = model.device()

    n_intermediate = 2**exp // 2  # intermediates per adjacent pair

    result: list[np.ndarray] = []
    for i in range(len(frames) - 1):
      I0 = self._frame_to_tensor(frames[i], device)
      I1 = self._frame_to_tensor(frames[i + 1], device)

      intermediate_tensors = self._make_inference(model, I0, I1, n_intermediate, scale)

      result.append(frames[i])
      for t in intermediate_tensors:
        result.append(self._tensor_to_frame(t))

    result.append(frames[-1])
    multiplier = 2**exp
    return result, multiplier
