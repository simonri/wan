from dataclasses import dataclass, field

from wan.configs.pipeline.base import PipelineConfig


@dataclass
class ServerArgs:
  pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

  output_path: str | None = "outputs/"
  input_save_path: str | None = "inputs/uploads"


_global_server_args: ServerArgs | None = None


def set_global_server_args(server_args: ServerArgs):
  global _global_server_args
  _global_server_args = server_args


def get_global_server_args() -> ServerArgs:
  if _global_server_args is None:
    raise ValueError("Global server args are not set")
  return _global_server_args
