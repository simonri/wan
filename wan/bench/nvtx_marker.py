import torch


class NVTXMarker:
  """Emit NVTX push/pop ranges around every (leaf or parent) module forward.

  Use as a context manager around the region you want labelled in an
  Nsight Systems trace. Range names use the dotted module name from
  ``model.named_modules()`` so blocks show up as e.g. ``blocks.20.self_attn``.
  """

  def __init__(self, model, name_filter=None, include_parents=True):
    self.model = model
    self.name_filter = name_filter.lower() if name_filter else None
    self.include_parents = include_parents
    self.handles = []

  def __enter__(self):
    for name, module in self.model.named_modules():
      if name == "":
        continue
      if not self.include_parents and any(module.children()):
        continue

      class_name = module.__class__.__name__
      if self.name_filter and self.name_filter not in name.lower() and self.name_filter not in class_name.lower():
        continue

      label = f"{name}::{class_name}"
      self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(label)))
      self.handles.append(module.register_forward_hook(self._make_post_hook()))
    return self

  def __exit__(self, exc_type, exc, tb):
    for handle in self.handles:
      handle.remove()

  def _make_pre_hook(self, label):
    def pre_hook(_module, _inputs):
      torch.cuda.nvtx.range_push(label)
    return pre_hook

  def _make_post_hook(self):
    def post_hook(_module, _inputs, _output):
      torch.cuda.nvtx.range_pop()
    return post_hook


def cuda_profiler_start():
  torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop():
  torch.cuda.cudart().cudaProfilerStop()
