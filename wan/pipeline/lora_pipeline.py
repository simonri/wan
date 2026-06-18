import time
from collections import defaultdict
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load_file

from wan.layers.lora.linear import replace_submodule, wrap_with_lora_layer
from wan.loader.utils import get_param_names_mapping
from wan.pipeline.base import PipelineBase
from wan.pipeline.lora_format_adapter import normalize_lora_state_dict
from wan.platform import get_local_torch_device
from wan.server_args import get_global_server_args


class LoRAPipeline(PipelineBase):
  lora_adapters: dict[str, dict[str, torch.Tensor]]  # nickname, target_weight_name -> weight
  loaded_adapter_paths: dict[str, str]  # nickname -> lora_path
  lora_layers: dict[str, Any]
  lora_layers_transformer_2: dict[str, Any]
  lora_rank: int | None
  lora_alpha: int | None
  lora_initialized: bool

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self.lora_adapters = defaultdict(dict)
    self.loaded_adapter_paths = {}

    self.lora_layers = {}
    self.lora_layers_transformer_2 = {}
    self.lora_rank = None
    self.lora_alpha = None
    self.lora_initialized = False

    self.device = get_local_torch_device()
    self.server_args = get_global_server_args()

  def convert_module_lora_layers(
    self, module: torch.nn.Module, module_name: str, target_lora_layers: dict[str, Any]
  ) -> int:
    converted_count = 0
    for name, layer in module.named_modules():
      lora_layer = wrap_with_lora_layer(
        layer,
        lora_rank=self.lora_rank,
        lora_alpha=self.lora_alpha,
      )
      if lora_layer is not None:
        target_lora_layers[name] = lora_layer
        replace_submodule(self.modules[module_name], name, lora_layer)
        converted_count += 1

    return converted_count

  def convert_to_lora_layers(self) -> None:
    """
    Convert the transformer to a LoRA transformer
    """
    if self.lora_initialized:
      return
    self.lora_initialized = True

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    converted_count = self.convert_module_lora_layers(
      self.modules["transformer"],
      "transformer",
      self.lora_layers,
    )
    print(f"Converted {converted_count} layers to LoRA layers in transformer ({time.perf_counter() - t0:.2f}s)")

    if "transformer_2" in self.modules and self.modules["transformer_2"] is not None:
      t0 = time.perf_counter()
      converted_count_2 = self.convert_module_lora_layers(
        self.modules["transformer_2"],
        "transformer_2",
        self.lora_layers_transformer_2,
      )
      print(f"Converted {converted_count_2} layers to LoRA layers in transformer_2 ({time.perf_counter() - t0:.2f}s)")

    print(f"== total convert_to_lora_layers: {time.perf_counter() - t_total:.2f}s ==")

  def load_lora_adapter(self, lora_path: str, lora_nickname: str):
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    raw_state_dict = safetensors_load_file(lora_path)
    lora_state_dict = normalize_lora_state_dict(raw_state_dict)
    t_read = time.perf_counter() - t0

    if lora_nickname in self.lora_adapters:
      self.lora_adapters[lora_nickname].clear()

    config = self.server_args.pipeline_config.dit_config.arch_config
    lora_param_names_mapping_fn = get_param_names_mapping(config.lora_param_names_mapping)

    t1 = time.perf_counter()

    for name, weight in lora_state_dict.items():
      name = name.removesuffix(".weight")
      name, _, _ = lora_param_names_mapping_fn(name)

      if name in self.lora_adapters[lora_nickname]:
        raise ValueError(f"Key {name} already exists in lora_adapters for {lora_nickname}!")
      self.lora_adapters[lora_nickname][name] = weight.to(self.device)
    t_to_device = time.perf_counter() - t1

    self.loaded_adapter_paths[lora_nickname] = lora_path
    print(
      f"Loaded LoRA adapter {lora_path}: "
      f"read+normalize={t_read:.2f}s  to_device={t_to_device:.2f}s  "
      f"total={time.perf_counter() - t_total:.2f}s"
    )

  def _apply_lora_to_layers(
    self,
    lora_layers: dict[str, Any],
    lora_nicknames: list[str],
    lora_paths: list[str | None],
    strengths: list[float],
    clear_existing: bool = False,
  ):
    """
    Apply LoRA weights to the given lora_layers.
    """
    if len(lora_paths) != len(lora_nicknames):
      raise ValueError("Number of lora_paths and lora_nicknames must be the same!")

    if len(strengths) != len(lora_paths):
      raise ValueError("Number of strengths and lora_paths must be the same!")

    t_apply = time.perf_counter()
    adapted_count = 0
    missing_layers_by_adapter = [[] for _ in lora_nicknames]
    applied_count_by_adapter = [0 for _ in lora_nicknames]
    for name, layer in lora_layers.items():
      # apply all LoRA adapters in order
      for idx, (nickname, path, lora_strength) in enumerate(zip(lora_nicknames, lora_paths, strengths, strict=True)):
        lora_A_name = name + ".lora_A"
        lora_B_name = name + ".lora_B"

        if lora_A_name in self.lora_adapters[nickname] and lora_B_name in self.lora_adapters[nickname]:
          inferred_rank = int(self.lora_adapters[nickname][lora_A_name].shape[0])
          alpha_key = name + ".alpha"
          if alpha_key in self.lora_adapters[nickname]:
            inferred_alpha = int(self.lora_adapters[nickname][alpha_key].item())
          else:
            inferred_alpha = inferred_rank

          layer.lora_rank = inferred_rank
          layer.lora_alpha = inferred_alpha

          layer.set_lora_weights(
            self.lora_adapters[nickname][lora_A_name],
            self.lora_adapters[nickname][lora_B_name],
            lora_path=path,
            strength=lora_strength,
            clear_existing=(clear_existing and idx == 0),  # Only clear on first LoRA
          )
          adapted_count += 1
          applied_count_by_adapter[idx] += 1
        else:
          missing_layers_by_adapter[idx].append(name)
          if idx == len(lora_nicknames) - 1:
            has_any_lora = any(
              name + ".lora_A" in self.lora_adapters[n] and name + ".lora_B" in self.lora_adapters[n]
              for n in lora_nicknames
            )
            if not has_any_lora:
              layer.disable_lora = True

    total_layers = len(lora_layers)
    for idx, path in enumerate(lora_paths):
      missing_layers = missing_layers_by_adapter[idx]
      if not missing_layers:
        continue
      applied_count = applied_count_by_adapter[idx]
      if applied_count == 0:
        print(f"LoRA adapter {path} did not match any LoRA layer")
      else:
        print(f"LoRA adapter {path} covers {applied_count}/{total_layers} layers")

    print(f"  _apply_lora_to_layers ({len(lora_nicknames)} adapters): {time.perf_counter() - t_apply:.2f}s")
    return adapted_count

  def _get_target_lora_layers(self, target: str):
    if target == "all":
      result = [("transformer", self.lora_layers)]
      if self.lora_layers_transformer_2:
        result.append(("transformer_2", self.lora_layers_transformer_2))
      return result, None
    elif target == "transformer":
      return [("transformer", self.lora_layers)], None
    elif target == "transformer_2":
      if not self.lora_layers_transformer_2:
        return [], "Transformer 2 not found in pipeline"
      return [("transformer_2", self.lora_layers_transformer_2)], None
    else:
      return [], "Invalid target!"

  def set_lora(
    self,
    lora_nicknames: None | list[str | None] = None,
    lora_paths: None | list[str | None] = None,
    targets: list[str] = None,
    strengths: list[float] = None,
  ):
    """
    Load LoRA into the pipeline and apply them to the specified transformer.

    Targets can be "all", "transformer", "transformer_2" etc.
    """
    if strengths is None:
      strengths = [1.0]
    if targets is None:
      targets = ["all"]
    if len(lora_paths) != len(lora_nicknames):
      raise ValueError("Number of lora_paths and lora_nicknames must be the same!")

    if len(strengths) != len(lora_nicknames):
      raise ValueError("Number of strengths and lora_nicknames must be the same!")

    if len(targets) != len(lora_nicknames):
      raise ValueError("Number of target and lora_nicknames must be the same!")

    t_set_lora = time.perf_counter()

    if not self.lora_initialized:
      self.convert_to_lora_layers()

    # load required adapters
    for nickname, path in zip(lora_nicknames, lora_paths, strict=True):
      if nickname not in self.lora_adapters and path is None:
        raise ValueError(f"Adapter {nickname} not found in pipeline. Please provide lora_path to load it!")

      should_load = False
      if path is not None:
        if nickname not in self.loaded_adapter_paths:
          should_load = True
        elif self.loaded_adapter_paths[nickname] != path:
          should_load = True

      if should_load:
        self.load_lora_adapter(path, nickname)

    # group by target to apply separately
    target_to_indices = {}
    for idx, tgt in enumerate(targets):
      if tgt not in target_to_indices:
        target_to_indices[tgt] = []
      target_to_indices[tgt].append(idx)

    adapted_count = 0
    for tgt, idx_list in target_to_indices.items():
      target_modules, error = self._get_target_lora_layers(tgt)
      if error:
        print(f"set_lora: {error}")
      if not target_modules:
        continue

      # apply LoRA to modules for this target
      for module_name, lora_layers_dict in target_modules:
        tgt_nicknames = [lora_nicknames[i] for i in idx_list]
        tgt_paths = [lora_paths[i] for i in idx_list]
        tgt_strengths = [strengths[i] for i in idx_list]

        count = self._apply_lora_to_layers(
          lora_layers_dict,
          tgt_nicknames,
          tgt_paths,
          tgt_strengths,
          clear_existing=True,
        )
        # TODO: we need to handle strength change here
        adapted_count += count

    print(f"== total set_lora: {time.perf_counter() - t_set_lora:.2f}s ==")

  def deactivate_lora_weights(self, target: str = "all") -> None:
    """Deactivate LoRA for the target transformer(s).

    Clears the active flag on each layer and disables the bypass path, so the
    forward runs the raw FP8 weights with no LoRA correction applied.
    """
    target_modules, error = self._get_target_lora_layers(target)
    if error:
      print(f"deactivate_lora_weights: {error}")
    if not target_modules:
      return

    t = time.perf_counter()
    for module_name, lora_layers_dict in target_modules:
      for layer in lora_layers_dict.values():
        if layer.merged:
          layer.unmerge_lora_weights()
        if not layer.disable_lora:
          layer.disable_lora = True

      print(f"LoRA weights deactivated for module {module_name}")

    print(f"== total deactivate_lora_weights({target}): {time.perf_counter() - t:.2f}s ==")
