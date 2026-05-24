from copy import deepcopy
from typing import Any, ClassVar

import torch

from wan.server_args import ServerArgs
from wan.stages.schedule_batch import Req


class StageDedupMixing:
  """
  Mixin for stage-local grouped-request dedupe.
  Handles only stage-local reuse.
  """

  deduplicated_output_fields: ClassVar[tuple[str, ...]] = ()
  deduplicated_tensor_tree_output_fields: ClassVar[tuple[str, ...]] = ()
  deduplicated_deepcopy_output_fields: ClassVar[tuple[str, ...]] = ()
  deduplicated_extra_tensor_tree_output_keys: ClassVar[tuple[str, ...]] = ()

  def run_grouped_requests(
    self,
    batches: list[Req],
    server_args: ServerArgs,
  ) -> list[Req]:
    """Run this stage for a group of independent requests.

    A grouped request is still a list of normal ``Req`` objects. The group
    boundary only gives a stage the opportunity to reduce duplicate work.
    Stages that do not opt in keep the single-request behavior by running
    ``self(batch, server_args)`` for every request.

    Full-stage dedup is declarative: declare the stage-owned output fields
    and return a stage-local fingerprint. Partial reuse belongs in a custom
    override, because the reusable unit is smaller than the whole stage.
    """
    if self.has_deduplicated_output_fields():
      return self.run_deduplicated_group(batches, server_args)

    return [self(batch, server_args) for batch in batches]

  def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs) -> Any:
    return id(batch)

  def run_deduplicated_group(
    self,
    batches: list[Req],
    server_args: ServerArgs,
    copy_outputs=None,
  ) -> list[Req]:
    """Run full-stage-equivalent requests once and fan out stage outputs."""
    if copy_outputs is None:
      copy_outputs = self.copy_deduplicated_outputs

    results: list[Req | None] = [None] * len(batches)

    for _, group in self._group_requests_by_fingerprint(
      batches, lambda batch: self.build_dedup_fingerprint(batch, server_args)
    ):
      first_index, first_batch = group[0]
      first_result = self(first_batch, server_args)
      results[first_index] = first_result

      for index, batch in group[1:]:
        copy_outputs(first_result, batch)
        results[index] = batch

    return [result for result in results if result is not None]

  def copy_deduplicated_outputs(self, src: "Req", dst: "Req") -> None:
    """Copy declared stage outputs from a computed request to a duplicate.

    ``deduplicated_output_fields`` uses shallow container copies and shares
    tensor references, which is the low-overhead path for read-only outputs
    such as embeddings. Tensor-tree fields recursively clone tensors.
    Deepcopy fields are for mutable request-local runtime objects, such as
    scheduler instances. Extra keys clone selected ``Req.extra`` entries
    without replacing the destination extra dict.
    """
    for field in self.deduplicated_output_fields:
      setattr(dst, field, self.copy_stage_output(getattr(src, field)))
    for field in self.deduplicated_tensor_tree_output_fields:
      setattr(dst, field, self.clone_tensor_tree(getattr(src, field)))
    for field in self.deduplicated_deepcopy_output_fields:
      setattr(dst, field, deepcopy(getattr(src, field)))
    for key in self.deduplicated_extra_tensor_tree_output_keys:
      if key in src.extra:
        dst.extra[key] = self.clone_tensor_tree(src.extra[key])

  @classmethod
  def copy_stage_output(cls, value):
    """Shallow-copy reusable containers while preserving tensor ownership."""
    if isinstance(value, list):
      return list(value)
    if isinstance(value, tuple):
      return tuple(value)
    if isinstance(value, dict):
      return dict(value)
    return value

  @classmethod
  def clone_tensor_tree(cls, value):
    """Recursively clone tensors in a small output tree."""
    if isinstance(value, torch.Tensor):
      return value.clone()
    if isinstance(value, list):
      return [cls.clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
      return tuple(cls.clone_tensor_tree(item) for item in value)
    if isinstance(value, dict):
      return {key: cls.clone_tensor_tree(item) for key, item in value.items()}
    return value

  @classmethod
  def has_deduplicated_output_fields(cls) -> bool:
    """Return whether this stage opts into base full-stage dedup."""
    return bool(
      cls.deduplicated_output_fields
      or cls.deduplicated_tensor_tree_output_fields
      or cls.deduplicated_deepcopy_output_fields
      or cls.deduplicated_extra_tensor_tree_output_keys
    )

  @staticmethod
  def _group_requests_by_fingerprint(
    batches: list[Req],
    fingerprint_fn,
  ) -> list[tuple[Any, list[tuple[int, Req]]]]:
    groups: dict[Any, list[tuple[int, Req]]] = {}
    for idx, batch in enumerate(batches):
      fingerprint = fingerprint_fn(batch)
      groups.setdefault(fingerprint, []).append((idx, batch))
    return list(groups.items())
