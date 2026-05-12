from abc import ABC, abstractmethod


class PipelineStage(ABC):
  def __init__(self):
    pass

  def __call__(self):
    result = self.forward()

    return result

  @abstractmethod
  def forward(self):
    raise NotImplementedError
