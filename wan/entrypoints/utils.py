import os
from copy import copy

from wan.stages.schedule_batch import Req


def _copy_req_for_output(
  req: Req,
):
  output_req = copy(req)
  output_req.sampling_params = copy(req.sampling_params)
  return output_req


def _with_output_index_suffix(output_file_name: str, output_index: int) -> str:
  base, ext = os.path.splitext(output_file_name)
  return f"{base}_{output_index}{ext}"


def normalize_output_seeds(
  seed: int | list[int],
  *,
  num_outputs_per_prompt: int,
  num_prompts: int = 1,
):
  if isinstance(seed, list):
    seeds = [int(item) for item in seed]
    total_outputs = num_outputs_per_prompt * num_prompts
    if len(seeds) == num_outputs_per_prompt:
      return seeds
    if len(seeds) == total_outputs:
      return seeds[0:num_outputs_per_prompt]
    raise ValueError(
      f"Seed list length must match num_outputs_per_prompt or total_outputs: {len(seeds)} != {num_outputs_per_prompt} or {total_outputs}"
    )

  base_seed = int(seed)
  return [base_seed + i for i in range(num_outputs_per_prompt)]


def expand_request_outputs(
  req: Req,
  *,
  num_prompts: int = 1,
):
  num_outputs = int(req.num_outputs_per_prompt)

  seeds = normalize_output_seeds(req.seed, num_outputs_per_prompt=num_outputs, num_prompts=num_prompts)

  if num_outputs == 1:
    req.seed = seeds[0]
    req.seeds = None
    req.generator = None

  expanded: list[Req] = []
  for output_idx, seed in enumerate(seeds):
    output_req = _copy_req_for_output(req)
    output_req.seed = seed
    output_req.num_outputs_per_prompt = 1
    output_req.seeds = None
    output_req.generator = None

    if output_req.image_path is not None:
      output_req.output_file_name = _with_output_index_suffix(req.output_file_name, output_idx)

    expanded.append(output_req)

  return expanded
