from dataclasses import dataclass, field


@dataclass
class ArchConfig:
  pass


@dataclass
class ModelConfig:
  # Every model config parameter can be categorized into either ArchConfig or everything else
  arch_config: ArchConfig = field(default_factory=ArchConfig)

  def __getattr__(self, name):
    # Only called if 'name' is not found in ModelConfig directly
    if hasattr(self.arch_config, name):
      return getattr(self.arch_config, name)
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

  def __getstate__(self):
    # Return a dictionary of attributes to pickle
    # Convert to dict and exclude any problematic attributes
    state = self.__dict__.copy()
    return state

  def __setstate__(self, state):
    # Restore instance attributes from the unpickled state
    self.__dict__.update(state)
