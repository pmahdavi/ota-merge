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

If the optimizer state (or an explicit pre‑conditioner tensor) is not
available we gracefully fall back to a scalar *weight* provided in the merge
configuration (mirroring LinearMerge).

To keep the implementation lightweight and practical we:

* Treat the diagonal pre‑conditioner as a *tensor* with the same shape as the
  model weight.
* Allow users to specify a path (local file) for each model that contains the
  pre‑conditioner in *safetensors* format.  The file must contain tensors
  whose names mirror the parameter names of the model weights (e.g.
  `model.layers.0.mlp.fc1.weight`).  Only the tensor matching the current
  `WeightInfo.name` is loaded on demand – keeping memory usage reasonable.
* If the path is **not** supplied or the tensor is missing, we revert to
  scalar weighting.  The default scalar weight is 1.0 so the behaviour is
  identical to simple averaging when no curvature information is available.

The method exposes two user‑facing parameter spaces:

Global parameters (specified once for the whole merge):
    • epsilon – numerical stability constant added to denominator (default
      1e‑8).

Per‑model tensor parameters:
    • weight – scalar weight (as in LinearMerge, default 1.0)
    • preconditioner_path – optional path to a *safetensors* file containing
      the Adam `exp_avg_sq` statistics for that model.

Usage example in YAML configuration:

```yaml
merge_method: ota
models:
  - model: specialist_model_math
    parameters:
      weight: 1.0
      preconditioner_path: "optimizer_states/math_exp_avg_sq.safetensors"
  - model: specialist_model_code
    parameters:
      weight: 1.0
      preconditioner_path: "optimizer_states/code_exp_avg_sq.safetensors"
# (global) parameters for the merge
parameters:
  epsilon: 1e-8
```
"""

from typing import Any, Dict, List, Optional

import logging
import os
from functools import lru_cache

import safetensors
import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

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


@lru_cache(maxsize=None)
def _open_safetensors(path: str):
    """Open a safetensors file once and cache the handle."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Pre‑conditioner file not found: {path}")
    return safetensors.safe_open(path, framework="pt", device="cpu")


# ---------------------------------------------------------------------------
# Helper for lazily loading torch optimizer checkpoints (state dicts)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _open_torch_opt_state(path: str):
    """Load a torch‑serialized optimizer checkpoint and extract exp_avg_sq.

    The file is expected to be the result of  `torch.save(optimizer.state_dict())`.
    The returned value is *not* the raw state_dict but a flat mapping
    {parameter_name: exp_avg_sq_tensor}.  This keeps the interface consistent
    with safetensors where a simple  key → tensor  mapping is used.
    """

    logger.debug("OTA: loading optimizer checkpoint %s", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Optimizer checkpoint not found: {path}")

    try:
        state_dict = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Unable to load optimizer checkpoint {path}: {e}")

    if not isinstance(state_dict, dict) or "state" not in state_dict:
        raise RuntimeError(
            f"Optimizer checkpoint {path} does not appear to contain a 'state' dict"
        )

    flat_map = {}
    for param_name, param_state in state_dict["state"].items():
        if isinstance(param_state, dict) and "exp_avg_sq" in param_state:
            flat_map[param_name] = param_state["exp_avg_sq"]
    if not flat_map:
        raise RuntimeError(
            f"Optimizer checkpoint {path} contains no 'exp_avg_sq' tensors"
        )
    return flat_map


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
    weight_info: WeightInfo

    # ---------------- Task boiler‑plate ----------------

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    # ---------------------------------------------------

    def _load_preconditioner(
        self, model: ModelReference, tensor: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Try to load the Adam `exp_avg_sq` tensor for *this* weight.

        Returns None if unavailable.
        """

        params = self.tensor_parameters[model]
        path = params.get("preconditioner_path")
        if not path:
            return None

        # Decide loader based on file extension.  Safetensors for *.safetensors,
        # otherwise treat as a torch optimizer checkpoint (*.pt / *.pth / *.bin).
        is_safetensors = path.lower().endswith(".safetensors")

        try:
            if is_safetensors:
                st = _open_safetensors(path)
                available_keys = st.keys()
            else:
                st = _open_torch_opt_state(path)
                available_keys = st.keys()
        except Exception as e:
            logger.warning(
                "OTA: failed to open pre‑conditioner file '%s' for model %s: %s",
                path,
                model,
                e,
            )
            return None

        # Determine the tensor key – we attempt exact match first, then some
        # common fallback naming conventions.
        if self.weight_info.name in available_keys:
            key = self.weight_info.name
        else:
            fallback_keys = [
                f"{self.weight_info.name}.exp_avg_sq",
                self.weight_info.name.replace("weight", "weight.exp_avg_sq"),
            ]
            key = next((k for k in fallback_keys if k in available_keys), None)
            if key is None:
                logger.warning(
                    "OTA: tensor '%s' not present in pre‑conditioner file '%s'",
                    self.weight_info.name,
                    path,
                )
                return None

        try:
            pc = (
                st.get_tensor(key) if is_safetensors else st[key]  # type: ignore[arg-type]
            )
        except Exception as e:
            logger.warning(
                "OTA: unable to read tensor '%s' from '%s': %s", key, path, e
            )
            return None

        # Ensure dtype and device alignment later; keep on CPU for now
        if pc.shape != tensor.shape:
            logger.warning(
                "OTA: pre‑conditioner shape mismatch for %s (got %s, expected %s)",
                self.weight_info.name,
                pc.shape,
                tensor.shape,
            )
            return None
        return pc

    # ---------------------------------------------------

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        models = list(tensors.keys())
        tensor_list = [tensors[m] for m in models]

        # Bring embeddings into agreement if needed (LLMs sometimes have slightly
        # different vocab sizes).
        rectify_embed_sizes(self.weight_info, tensor_list)

        # Sanity check – shapes must be identical after rectification.
        unique_shapes = {t.shape for t in tensor_list}
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )

        # Convert the list to a single stacked tensor for efficient math.
        stacked = torch.stack(tensor_list, dim=0)

        # Move to appropriate device/dtype once to avoid repeated transfers.
        device = stacked.device
        dtype = stacked.dtype

        # Build per‑model, per‑element weighting tensors.
        weight_tensors = []
        for i, m in enumerate(models):
            # Try to load diagonal pre‑conditioner.
            pc = self._load_preconditioner(m, stacked[i])
            if pc is not None:
                pc = pc.to(device=device, dtype=dtype)
                # Adam curvature proxy
                precond = torch.sqrt(pc + self.epsilon)
                weight_tensor = precond + self.epsilon  # (P_i + ε)
            else:
                # If a preconditioner path was provided but loading failed, emit a
                # reminder that we are falling back to the scalar weight.
                if self.tensor_parameters[m].get("preconditioner_path"):
                    logger.warning(
                        "OTA: falling back to scalar weight for %s because pre‑conditioner could not be loaded",
                        m,
                    )
                # Fallback: scalar weight.
                scalar_w = self.tensor_parameters[m].get("weight", 1.0)
                wt = torch.tensor(scalar_w, dtype=dtype, device=device)
                # broadcast to full shape
                while wt.dim() < stacked[i].dim():
                    wt = wt.unsqueeze(-1)
                weight_tensor = wt.expand_as(stacked[i]) + self.epsilon  # (w_i + ε)

            weight_tensors.append(weight_tensor)

        weights_stack = torch.stack(weight_tensors, dim=0)

        # Compute OTA merge: element‑wise weighted average.
        numerator = (weights_stack * stacked).sum(dim=0)
        # Denominator already contains  Σ (P_i + ε)  ⇒  Σ P_i + n·ε
        denominator = weights_stack.sum(dim=0)
        merged = numerator / denominator
        return merged

    # ---------------------------------------------------

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


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
        return [ConfigParameterDef(name="epsilon", required=False, default_value=1e-8)]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        # `weight` is optional (defaults to 1.0).  `preconditioner_path` is optional.
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
            ConfigParameterDef(name="preconditioner_path", required=False),
        ]

    # ---- Task creation ----

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return OTAMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            epsilon=parameters.get("epsilon", 1e-8),
            weight_info=output_weight,
        ) 