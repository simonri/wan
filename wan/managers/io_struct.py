"""Wire types between HTTP process and the GPU worker process.

Mirrors sglang's io_struct.py role. Keep these dataclasses simple and pickle-friendly;
they cross a ZMQ socket via send_pyobj/recv_pyobj.
"""

from dataclasses import dataclass, field

from wan.entrypoints.protocol import LoRAItem
from wan.stages.schedule_batch import Req


@dataclass
class BatchGenerateReq:
  """HTTP -> Scheduler. Carries the model-level Req plus postproc knobs that
  belong to the response shape, not the diffusion call itself.
  """

  job_id: str
  req: Req
  loras: list[LoRAItem] = field(default_factory=list)
  enable_frame_interpolation: bool = False
  frame_interpolation_exp: int = 1
  frame_interpolation_scale: float = 1.0
  crf: int = 23


@dataclass
class BatchGenerateOutput:
  """Scheduler -> HTTP. file_paths is None when error is set."""

  job_id: str
  file_paths: list[str] | None = None
  error: str | None = None
  inference_time_s: float | None = None
  num_outputs: int | None = None
  peak_memory_mb: float | None = None
