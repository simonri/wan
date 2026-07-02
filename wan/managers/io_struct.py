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
  # Streaming: encode fMP4 in the worker and return the bytes directly,
  # skipping the disk MP4 + remux + ffprobe round-trip on the HTTP side.
  return_fmp4: bool = False
  # Set False to skip writing the MP4 file entirely (streaming-only jobs).
  save_file: bool = True
  fmp4_preset: str = "veryfast"


@dataclass
class BatchGenerateOutput:
  """Scheduler -> HTTP. file_paths is None when error is set."""

  job_id: str
  file_paths: list[str] | None = None
  error: str | None = None
  inference_time_s: float | None = None
  num_outputs: int | None = None
  peak_memory_mb: float | None = None
  # Populated when the request set return_fmp4 (first output only).
  fmp4_init: bytes | None = None
  fmp4_media: bytes | None = None
  duration_s: float | None = None
