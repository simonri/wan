import torch


def get_local_torch_device() -> torch.device:
  return CudaPlatform.get_local_torch_device()


class CudaPlatform:
  @classmethod
  def get_local_torch_device(cls) -> torch.device:
    return torch.device("cuda:0")

  @classmethod
  def get_available_gpu_memory(
    cls,
    device_id: int = 0,
  ) -> float:
    free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)
    return free_gpu_memory / (1 << 30)
