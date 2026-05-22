import re
from collections.abc import Callable
from typing import Any


def get_param_names_mapping(
  mapping_dict: dict[str, str | tuple[str, int, int]],
) -> Callable[[str], tuple[str, Any, Any]]:
  """
  Create a mapping func that transforms param names using regex patterns.
  """

  def mapping_fn(name: str) -> tuple[str, Any, Any]:
    # support chained conversions, e.g.:
    # transformer.xxx.lora_down -> xxx.lora_down -> xxx.proj_down
    merge_index = None
    total_split_params = None
    max_steps = max(8, len(mapping_dict) * 2)
    applied_patterns: set[str] = set()
    visited_names: set[str] = {name}

    for _ in range(max_steps):
      transformed = False
      for pattern, replacement in mapping_dict.items():
        # avoid re-applying the same rule on its own output
        if pattern in applied_patterns:
          continue
        if re.match(pattern, name) is None:
          continue

        curr_merge_index = None
        curr_total_split_params = None
        if isinstance(replacement, tuple):
          curr_merge_index = replacement[1]
          curr_total_split_params = replacement[2]
          replacement = replacement[0]

        new_name = re.sub(pattern, replacement, name)

        if new_name != name:
          if curr_merge_index is not None:
            merge_index = curr_merge_index
            total_split_params = curr_total_split_params

          name = new_name
          applied_patterns.add(pattern)
          if name in visited_names:
            transformed = False
            break
          visited_names.add(name)
          transformed = True
          break

      if not transformed:
        break

    return name, merge_index, total_split_params

  return mapping_fn
