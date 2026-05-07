import collections

import torch


class LayerTimer:
  def __init__(self, model, name_filter=None, include_parents=False):
    self.model = model
    self.name_filter = name_filter.lower() if name_filter else None
    self.include_parents = include_parents
    self.handles = []
    self.starts = collections.defaultdict(list)
    self.timings = collections.OrderedDict()
    self.module_classes = {}

  def __enter__(self):
    for name, module in self.model.named_modules():
      if name == "":
        continue
      if not self.include_parents and any(module.children()):
        continue

      class_name = module.__class__.__name__
      if self.name_filter and self.name_filter not in name.lower() and self.name_filter not in class_name.lower():
        continue

      self.module_classes[name] = class_name
      self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
      self.handles.append(module.register_forward_hook(self._make_post_hook(name)))

    return self

  def __exit__(self, exc_type, exc, tb):
    for handle in self.handles:
      handle.remove()

  def _make_pre_hook(self, name):
    def pre_hook(_module, _inputs):
      start = torch.cuda.Event(enable_timing=True)
      start.record()
      self.starts[name].append(start)

    return pre_hook

  def _make_post_hook(self, name):
    def post_hook(_module, _inputs, _output):
      end = torch.cuda.Event(enable_timing=True)
      end.record()
      start = self.starts[name].pop()
      self.timings.setdefault(name, []).append((start, end))

    return post_hook

  def results_ms(self):
    torch.cuda.synchronize()
    rows = []
    for name, events in self.timings.items():
      total_ms = sum(start.elapsed_time(end) for start, end in events)
      rows.append((total_ms, len(events), self.module_classes[name], name))
    return sorted(rows, reverse=True)
