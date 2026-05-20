from dataclasses import dataclass, field

from wan.configs.pipeline.base import PipelineConfig


@dataclass
class ServerArgs:
  pipeline_config: PipelineConfig = field(default_factory=PipelineConfig, repr=False)

  output_path: str | None = "outputs/"
  input_save_path: str | None = "inputs/uploads"
