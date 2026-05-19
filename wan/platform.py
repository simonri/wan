import torch


def get_local_torch_device() -> torch.device:
  return CudaPlatform.get_local_torch_device()


class CudaPlatform:
  @classmethod
  def get_local_torch_device(cls) -> torch.device:
    return torch.device("cuda:0")
