from dataclasses import dataclass, field
from uuid import uuid4

from wan.configs.pipeline.base import PipelineConfig


@dataclass
class ServerArgs:
  pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

  output_path: str | None = "outputs"
  input_save_path: str | None = "inputs/uploads"
  text_embed_cache_dir: str | None = "cache/text_embeds"

  # HTTP server bind. uvicorn picks these up directly.
  host: str = "0.0.0.0"
  port: int = 8001

  # IPC endpoints between HTTP process and the GPU worker process.
  # sglang convention: the long-lived (HTTP) side binds, the worker connects.
  # Initialized lazily in launch_server() so both processes share the same UUID
  # after fork via ServerArgs copy.
  scheduler_input_ipc_name: str = ""
  tokenizer_ipc_name: str = ""

  def init_ipc_names(self) -> None:
    """Assign random IPC paths. Call once on the parent before forking the worker."""
    token = uuid4().hex[:12]
    self.scheduler_input_ipc_name = f"ipc:///tmp/wan-scheduler-input-{token}.ipc"
    self.tokenizer_ipc_name = f"ipc:///tmp/wan-tokenizer-{token}.ipc"


_global_server_args: ServerArgs | None = None


def set_global_server_args(server_args: ServerArgs):
  global _global_server_args
  _global_server_args = server_args


def get_global_server_args() -> ServerArgs:
  if _global_server_args is None:
    raise ValueError("Global server args are not set")
  return _global_server_args
