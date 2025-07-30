# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1
"""OTA (Optimization Trajectory Aware) merging method.

This implementation follows the high‑level formulation proposed in the
"Harnessing Optimization Dynamics for Efficient Geometry‑Informed Model
Merging" paper (see context in conversation).

Key idea:  For every model that is merged we attempt to load the diagonal
pre‑conditioner (the exponential moving average of squared gradients –
`exp_avg_sq` in Adam terminology).  Given the per‑parameter pre‑conditioner
vector \(P_\tau \approx \sqrt{v}\), OTA merging computes a merge that is
*weighted element‑wise* by these curvature estimates:

w_merged = ( Σ_τ  P_τ * w_τ ) / ( Σ_τ P_τ )

This method requires that every model being merged has a corresponding
preconditioner file specified via the `preconditioner_path` parameter. It does
not support fallback to simple weighting.

To keep the implementation lightweight and practical we:

* Treat the diagonal pre‑conditioner as a *tensor* with the same shape as the
  model weight.
* A path must be specified for each model that contains the
  pre‑conditioner in *safetensors* format.  The file must contain tensors
  whose names mirror the parameter names of the model weights (e.g.
  `model.layers.0.mlp.fc1.weight`).  Only the tensor matching the current
  `WeightInfo.name` is loaded on demand – keeping memory usage reasonable.

The method exposes two user‑facing parameter spaces:

Global parameters (specified once for the whole merge):
    • epsilon – numerical stability constant added to denominator (default
      1e‑8).
    • normalise – how to normalize preconditioner tensors before use:
      "none" (default), "global".
    • precond_threshold – element-wise threshold for preconditioner values.
      Elements with precond values below this will be masked out (set to zero).

Per‑model tensor parameters:
    • preconditioner_path – path to a *safetensors* file containing
      the Adam `exp_avg_sq` statistics for that model.

Usage example in YAML configuration:

```yaml
merge_method: ota
models:
  - model: specialist_model_math
    parameters:
      preconditioner_path: "optimizer_states/math_exp_avg_sq.safetensors"
  - model: specialist_model_code
    parameters:
      preconditioner_path: "optimizer_states/code_exp_avg_sq.safetensors"
# (global) parameters for the merge
parameters:
  epsilon: 1e-8
  normalise: global        # none | global
  power: 0.25        # v^0.25
  precond_threshold: 1e-4  # Optional: mask elements where precond < threshold
```
"""

from typing import Any, Dict, List, Optional

import logging
import os
import json # Added for model manifest
from functools import lru_cache

import safetensors
import torch
from pydantic import PrivateAttr
from typing_extensions import override
from huggingface_hub import hf_hub_download, HfApi

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes
from mergekit.options import MergeOptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NOTE: OTA supports two formats for curvature (pre‑conditioner) files:
#   1.  *.safetensors — a file that directly stores one tensor per parameter.
#   2.  *.pt / *.pth / *.bin  — a torch‑serialized optimizer checkpoint that
#       contains a dictionary with keys  "state"  and  "param_groups".  The
#       2‑level  state[param_name]['exp_avg_sq']  entry holds the diagonal of
#       the Adam second‑moment estimate.  We lazily convert that structure
#       into a mapping {param_name → tensor} and cache it so that every file
#       is deserialized at most once.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper for lazily loading torch optimizer checkpoints (state dicts)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The graph Task performing OTA merge for **one** tensor
# ---------------------------------------------------------------------------


class OTAMergeTask(Task[torch.Tensor]):
    """Graph task that merges a single tensor according to OTA.

    It receives the original model tensors via `gather_tensors` and merges
    them element‑wise using curvature information derived from the Adam
    pre‑conditioner if available.
    """

    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    epsilon: float
    normalise_mode: str
    weight_info: WeightInfo
    merge_options: Optional[MergeOptions] = None
    precond_power: float
    precond_threshold: Optional[float] = None
    base_model_ref: Optional[ModelReference] = None
    fallback_to_base: bool = False
    _preconditioner_loader: "PreconditionerLoader" = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the preconditioner loader after the model is validated."""
        self._preconditioner_loader = PreconditionerLoader(self.weight_info)

    # ---------------- Task boiler‑plate ----------------

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    # ---------------------------------------------------

    def _load_preconditioner(
        self, model: ModelReference, tensor: torch.Tensor, path: str
    ) -> torch.Tensor:
        """Load the Adam `exp_avg_sq` tensor for *this* weight.
        Raises an error if loading fails.
        """
        return self._preconditioner_loader.load(model, tensor, path)

    def _prepare_and_stack_tensors(
        self, tensor_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Rectify, validate, and stack input tensors in fp32."""
        rectify_embed_sizes(self.weight_info, tensor_list)

        tensor_list_fp32 = [t.to(torch.float32) for t in tensor_list]

        unique_shapes = {t.shape for t in tensor_list_fp32}
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )

        return torch.stack(tensor_list_fp32, dim=0)

    def _get_and_load_preconditioner(
        self, model: ModelReference, tensor: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Get path and load preconditioner tensor."""
        params = self.tensor_parameters[model]
        path = params.get("preconditioner_path")
        if not path:
            raise ValueError(
                f"OTA merge requires a 'preconditioner_path' for all models. Model '{model}' is missing one."
            )
        pc = self._load_preconditioner(model, tensor, path)
        return pc.to(device=device, dtype=torch.float32)

    def _apply_preconditioner_mask(
        self,
        pc: torch.Tensor,
        model: ModelReference,
        specialist_tensor: Optional[torch.Tensor] = None,
        base_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Apply a threshold to the preconditioner and return a mask."""
        if self.precond_threshold is None:
            return None

        threshold = float(self.precond_threshold)
        log_message = ""

        if (
            self.base_model_ref
            and base_tensor is not None
            and specialist_tensor is not None
        ):
            # New Fisher-based masking logic
            task_vector = specialist_tensor.to(torch.float32) - base_tensor.to(
                torch.float32
            )
            score = pc * torch.pow(task_vector, 2)
            mask = score >= threshold
            log_message = "Applying Fisher-based precond threshold"
        else:
            # Original masking logic
            mask = pc >= threshold
            log_message = "Applying precond threshold"

        num_masked = (~mask).sum().item()
        total_elements = pc.numel()
        mask_percentage = (
            (num_masked / total_elements) * 100 if total_elements > 0 else 0
        )
        logger.info(
            "OTA Task (%s): Model %s - %s %.4e. Masked %d/%d elements (%.2f%%)",
            self.weight_info.name,
            model,
            log_message,
            threshold,
            num_masked,
            total_elements,
            mask_percentage,
        )
        return mask

    def _transform_and_scale_preconditioner(
        self, pc: torch.Tensor, model: ModelReference
    ) -> torch.Tensor:
        """Apply power transformation and scaling factor in fp32."""
        p = float(self.precond_power)
        if p <= 0:
            raise ValueError("OTA power must be > 0")
        precond = torch.pow(pc, p)  # Remains fp32

        params = self.tensor_parameters[model]
        scaling_factor = float(params.get("scaling_factor", 1.0))
        if scaling_factor != 1.0:
            precond = precond * scaling_factor
        return precond

    def _normalize_preconditioner(self, precond: torch.Tensor) -> torch.Tensor:
        """Normalize the preconditioner tensor in fp32."""
        if self.normalise_mode == "global":
            scale = precond.mean()  # Already fp32, .float() is redundant
            return precond / (scale + self.epsilon)
        if self.normalise_mode != "none":
            raise ValueError(f"Unknown normalisation mode for OTA: {self.normalise_mode}")
        return precond

    def _calculate_single_weight_tensor(
        self,
        model: ModelReference,
        tensor: torch.Tensor,
        device: torch.device,
        base_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate the weighting tensor for a single model in fp32."""
        pc = self._get_and_load_preconditioner(model, tensor, device)  # returns fp32
        mask = self._apply_preconditioner_mask(
            pc, model, specialist_tensor=tensor, base_tensor=base_tensor
        )  # returns boolean mask

        precond = self._transform_and_scale_preconditioner(pc, model)  # returns fp32
        precond = self._normalize_preconditioner(precond)  # returns fp32

        weight_tensor = precond + self.epsilon  # fp32

        if mask is not None:
            # mask is boolean, multiplication promotes it to float32
            weight_tensor = weight_tensor * mask

        return weight_tensor  # Already fp32, .float() is redundant

    def _compute_weighting_tensors(
        self,
        models: List[ModelReference],
        stacked_tensors: torch.Tensor,
        base_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute element-wise weighting tensors for all models."""
        logger.info(
            "OTA Task (%s): Normalisation mode set to '%s'.",
            self.weight_info.name,
            self.normalise_mode,
        )

        weight_tensors = [
            self._calculate_single_weight_tensor(
                model=m,
                tensor=stacked_tensors[i],
                device=stacked_tensors.device,
                base_tensor=base_tensor,
            )
            for i, m in enumerate(models)
        ]

        return torch.stack(weight_tensors, dim=0)

    def _perform_weighted_merge(
        self,
        stacked_tensors: torch.Tensor,
        weights_stack: torch.Tensor,
        base_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform the element-wise weighted merge with a fallback for fully masked elements."""
        numerator = (weights_stack * stacked_tensors).sum(dim=0)
        denominator = weights_stack.sum(dim=0)

        # The denominator should not be zero due to the added epsilon,
        # making this division safe.
        merged_tensor = numerator / denominator

        if self.precond_threshold is None:
            return merged_tensor

        # Fallback logic for elements where all models were masked by the threshold
        n_models = weights_stack.shape[0]
        # A small tolerance is added to account for floating point inaccuracies.
        fully_masked_mask = denominator <= (n_models * self.epsilon * 1.1)

        if fully_masked_mask.any():
            n_masked_elements = fully_masked_mask.sum().item()
            total_elements = denominator.numel()
            percentage = (
                (n_masked_elements / total_elements * 100) if total_elements > 0 else 0
            )

            fallback_value: torch.Tensor
            log_message: str

            if self.fallback_to_base and base_tensor is not None:
                fallback_value = base_tensor.to(
                    dtype=merged_tensor.dtype, device=merged_tensor.device
                )
                log_message = "Falling back to base model value for these elements."
            else:
                fallback_value = stacked_tensors.mean(dim=0)
                log_message = "Falling back to simple average for these elements."

            logger.warning(
                "OTA Task (%s): %d/%d elements (%.2f%%) have all models masked due to precond_threshold=%.1e. %s",
                self.weight_info.name,
                n_masked_elements,
                total_elements,
                percentage,
                self.precond_threshold,
                log_message,
            )

            # For the fully masked elements, use the chosen fallback value.
            merged_tensor = torch.where(
                fully_masked_mask, fallback_value, merged_tensor
            )

        return merged_tensor

    # ---------------------------------------------------

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        logger.info("OTA Task: Processing tensor '%s'", self.weight_info.name)

        if self.base_model_ref:
            if self.base_model_ref not in tensors:
                raise ValueError(
                    f"Base model '{self.base_model_ref}' tensor not found for {self.weight_info.name}"
                )
            base_tensor = tensors[self.base_model_ref]
            models_to_merge = [m for m in tensors if m != self.base_model_ref]
        else:
            base_tensor = None
            models_to_merge = list(tensors.keys())

        tensor_list = [tensors[m] for m in models_to_merge]

        if not tensor_list:
            # This can happen if the only model provided is the base model.
            if base_tensor is not None:
                logger.warning(
                    "OTA Task (%s): No models to merge (only a base model was provided). Returning the base model's tensor without changes.",
                    self.weight_info.name,
                )
                return base_tensor.to(tensors[self.base_model_ref].dtype)
            raise ValueError(f"No tensors provided to merge for {self.weight_info.name}")

        # Get original dtype from the first model tensor
        original_dtype = tensor_list[0].dtype

        stacked_tensors = self._prepare_and_stack_tensors(tensor_list)
        weights_stack = self._compute_weighting_tensors(
            models_to_merge, stacked_tensors, base_tensor
        )

        merged = self._perform_weighted_merge(stacked_tensors, weights_stack, base_tensor)

        merged_stats = merged.float()
        logger.info(
            "OTA Task (%s): Final merged tensor stats: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
            self.weight_info.name,
            merged_stats.min(),
            merged_stats.max(),
            merged_stats.mean(),
            merged_stats.std(),
        )
        return merged.to(original_dtype)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class PreconditionerLoader:
    """A utility class to handle loading of preconditioner tensors from various sources."""

    def __init__(self, weight_info: WeightInfo):
        self.weight_info = weight_info

    @lru_cache(maxsize=None)
    def _get_preconditioner_file(self, path: str):
        """
        Opens a preconditioner file (either safetensors or torch checkpoint)
        and returns a handle or a parsed dictionary. This method caches the result.
        """
        path = self._download_from_hf_hub(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pre-conditioner file not found: {path}")

        if path.lower().endswith(".safetensors"):
            return safetensors.safe_open(path, framework="pt", device="cpu")

        try:
            state_dict = torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Unable to load optimizer checkpoint {path}: {e}")

        if not isinstance(state_dict, dict) or "state" not in state_dict:
            raise RuntimeError(
                f"Optimizer checkpoint {path} does not appear to contain a 'state' dict"
            )

        flat_map = {
            param_name: param_state["exp_avg_sq"]
            for param_name, param_state in state_dict["state"].items()
            if isinstance(param_state, dict) and "exp_avg_sq" in param_state
        }

        if not flat_map:
            raise RuntimeError(
                f"Optimizer checkpoint {path} contains no 'exp_avg_sq' tensors"
            )
        return flat_map

    def load(self, model: ModelReference, tensor: torch.Tensor, path: str) -> torch.Tensor:
        """Load the preconditioner tensor for a given model and tensor."""
        try:
            preconditioner_file = self._get_preconditioner_file(path)
            is_safetensors = not isinstance(preconditioner_file, dict)
            available_keys = (
                preconditioner_file.keys()
                if is_safetensors
                else list(preconditioner_file.keys())
            )
        except Exception as e:
            raise RuntimeError(
                f"OTA Task ({self.weight_info.name}): Failed to open/load preconditioner file '{path}' for model {model}. Error: {e}"
            ) from e

        key = self._get_preconditioner_tensor_key(available_keys, path)
        pc = self._load_tensor_from_file(preconditioner_file, key, path)

        if pc.shape != tensor.shape:
            raise ValueError(
                f"OTA Task ({self.weight_info.name}): Preconditioner shape mismatch for {self.weight_info.name} (model {model}). Got {pc.shape}, expected {tensor.shape}. File: {path}"
            )
        return pc

    def _download_from_hf_hub(self, path: str) -> str:
        """Download a file from Hugging Face Hub if it doesn't exist locally."""
        if os.path.exists(path) or not ("/" in path and len(path.split("/")) > 2):
            return path
        try:
            parts = path.split("/")
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
            logger.info(f"Downloading preconditioner from {repo_id} - {filename}")
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            raise RuntimeError(f"Failed to download preconditioner from HF: {e}") from e

    def _get_preconditioner_tensor_key(
        self, available_keys: List[str], path: str
    ) -> str:
        """Determine the tensor key with fallbacks."""
        if self.weight_info.name in available_keys:
            return self.weight_info.name

        fallback_keys = [
            f"{self.weight_info.name}.exp_avg_sq",
            self.weight_info.name.replace("weight", "weight.exp_avg_sq"),
        ]
        key = next((k for k in fallback_keys if k in available_keys), None)

        if key is None:
            raise KeyError(
                f"OTA Task ({self.weight_info.name}): Tensor name '{self.weight_info.name}' (or fallbacks) not present in preconditioner file '{path}'. Available keys: {list(available_keys)}"
            )
        return key

    def _load_tensor_from_file(
        self, preconditioner_file, key: str, path: str
    ) -> torch.Tensor:
        """Load a tensor from a file, given the key and file type."""
        try:
            if not isinstance(preconditioner_file, dict):
                return preconditioner_file.get_tensor(key)
            return preconditioner_file[key]
        except Exception as e:
            raise RuntimeError(
                f"OTA Task ({self.weight_info.name}): Unable to read tensor '{key}' from '{path}': {e}"
            ) from e


# ---------------------------------------------------------------------------
# MergeMethod wrapper
# ---------------------------------------------------------------------------


class OTAMerge(MergeMethod):
    """Optimization Trajectory Aware (OTA) model merging method."""

    # ---- Metadata ----

    def name(self) -> str:
        return "ota"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Optimization Trajectory Aware (OTA)"

    @override
    def reference_url(self) -> Optional[str]:
        # No official URL yet (anonymous submission).  Replace when published.
        return None

    # ---- Parameter schemas ----

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="epsilon", required=False, default_value=1e-8),
            ConfigParameterDef(
                name="normalise",
                required=False,
                default_value="none",
                description="How to normalize preconditioner tensors before use: 'none', 'global'",
            ),
            ConfigParameterDef(
                name="power",
                required=False,
                default_value=0.5,
                description="Exponent p used for curvature weighting: weight ∝ v^p. Typical values: 0.25, 0.5, 1.0. Must be > 0.",
            ),
            ConfigParameterDef(
                name="precond_threshold",
                required=False,
                default_value=None,
                description="Element-wise threshold for preconditioner values. Elements with precond values below this threshold will be masked out (set to zero). Applied to raw preconditioner values before power transformation.",
            ),
            ConfigParameterDef(
                name="fallback_to_base",
                required=False,
                default_value=False,
                description="If true, fully masked elements will revert to the base model's value instead of a simple average.",
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        # `preconditioner_path` is required.
        return [
            ConfigParameterDef(name="preconditioner_path", required=True),
            ConfigParameterDef(name="scaling_factor", required=False, default_value=1.0, description="Optional multiplicative factor applied to this model's preconditioner to allow intensity normalisation across models."),
        ]

    # ---- Task creation ----

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        merge_options: Optional[MergeOptions] = None
    ) -> Task:
        task = OTAMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            epsilon=parameters.get("epsilon", 1e-8),
            normalise_mode=str(parameters.get("normalise", "none")).lower(),
            weight_info=output_weight,
            merge_options=merge_options,
            precond_power=parameters.get("power", 0.5),
            precond_threshold=parameters.get("precond_threshold"),
            base_model_ref=base_model,
            fallback_to_base=parameters.get("fallback_to_base", False),
        )
        return task 