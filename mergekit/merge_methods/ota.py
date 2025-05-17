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
    • normalise – how to normalize preconditioner tensors before use:
      "none" (default), "global", "layer", "inv", "inv-global".

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
  normalise: global        # none | global | layer | inv | inv-global
  clip_factor: 5.0       # For "inv" mode, clips weights at N × median. None/null to disable.
  power: 0.25        # v^0.25 then inverted
```
"""

from typing import Any, Dict, List, Optional

import logging
import os
import json # Added for model manifest
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
    normalise_mode: str
    weight_info: WeightInfo
    merge_options: Optional[MergeOptions] = None
    out_path_for_detailed_logs: Optional[str] = None
    clip_factor: Optional[float] = None
    precond_power: float

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
        logger.debug(
            "OTA Task (%s): Attempting to load preconditioner for model %s, tensor %s",
            self.weight_info.name,
            model,
            self.weight_info.name,
        )
        params = self.tensor_parameters[model]
        path = params.get("preconditioner_path")
        if not path:
            logger.debug(
                "OTA Task (%s): No preconditioner_path specified for model %s.",
                self.weight_info.name,
                model,
            )
            return None

        logger.debug(
            "OTA Task (%s): Preconditioner path for model %s is %s.",
            self.weight_info.name,
            model,
            path,
        )

        # Decide loader based on file extension.  Safetensors for *.safetensors,
        # otherwise treat as a torch optimizer checkpoint (*.pt / *.pth / *.bin).
        is_safetensors = path.lower().endswith(".safetensors")

        try:
            if is_safetensors:
                st = _open_safetensors(path)
                available_keys = st.keys()
                logger.debug("OTA Task (%s): Successfully opened safetensor: %s", self.weight_info.name, path)
            else:
                st = _open_torch_opt_state(path)
                available_keys = st.keys()
                logger.debug("OTA Task (%s): Successfully loaded torch optimizer state: %s", self.weight_info.name, path)
        except Exception as e:
            logger.warning(
                "OTA Task (%s): Failed to open/load preconditioner file '%s' for model %s. Error: %s",
                self.weight_info.name,
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
                    "OTA Task (%s): Tensor name '%s' (or fallbacks) not present in preconditioner file '%s'. Available keys: %s",
                    self.weight_info.name,
                    self.weight_info.name,
                    path,
                    list(available_keys)
                )
                return None
            else:
                logger.debug(
                    "OTA Task (%s): Found key '%s' for tensor '%s' in preconditioner file '%s'",
                    self.weight_info.name,
                    key,
                    self.weight_info.name,
                    path,
                )

        try:
            pc = (
                st.get_tensor(key) if is_safetensors else st[key]  # type: ignore[arg-type]
            )
        except Exception as e:
            logger.warning(
                "OTA Task (%s): Unable to read tensor '%s' from '%s': %s", self.weight_info.name, key, path, e
            )
            return None

        # Ensure dtype and device alignment later; keep on CPU for now
        if pc.shape != tensor.shape:
            logger.warning(
                "OTA Task (%s): Preconditioner shape mismatch for %s (model %s). Got %s, expected %s. File: %s",
                self.weight_info.name,
                self.weight_info.name,
                model,
                pc.shape,
                tensor.shape,
                path,
            )
            return None
        logger.debug(
            "OTA Task (%s): Successfully loaded preconditioner tensor '%s' for model %s from %s with shape %s, dtype %s.",
            self.weight_info.name,
            key,
            model,
            path,
            pc.shape,
            pc.dtype,
        )
        return pc

    # ---------------------------------------------------

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        logger.info("OTA Task: Processing tensor '%s'", self.weight_info.name)
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
        logger.debug(
            "OTA Task (%s): Stacked model tensors on device '%s' with dtype '%s'. Shape: %s",
            self.weight_info.name,
            device,
            dtype,
            stacked.shape,
        )

        # Build per‑model, per‑element weighting tensors.
        weight_tensors = []
        
        norm_mode = self.normalise_mode
        logger.info(
            "OTA Task (%s): Normalisation mode set to '%s'.",
            self.weight_info.name,
            norm_mode
        )

        for i, m in enumerate(models):
            # Try to load diagonal pre‑conditioner.
            pc = self._load_preconditioner(m, stacked[i])
            if pc is not None:
                original_pc_dtype = pc.dtype
                logger.debug(
                    "OTA Task (%s): Model %s - Preconditioner tensor loaded. Original dtype: %s. Target dtype: %s",
                    self.weight_info.name,
                    m,
                    original_pc_dtype,
                    dtype,
                )
                pc_stats_raw = pc.float() # Calculate stats in float32 for stability if original is low precision
                logger.debug(
                    "OTA Task (%s): Model %s - Raw PC stats (float32): Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
                    self.weight_info.name, m, pc_stats_raw.min(), pc_stats_raw.max(), pc_stats_raw.mean(), pc_stats_raw.std()
                )

                pc = pc.to(device=device, dtype=dtype)
                logger.debug(
                    "OTA Task (%s): Model %s - PC tensor converted to device '%s', dtype '%s'. Shape: %s",
                    self.weight_info.name, m, pc.device, pc.dtype, pc.shape
                )
                
                # Adam curvature proxy
                p = float(self.precond_power)
                if p <= 0:
                    raise ValueError("OTA power must be > 0")
                # cast to fp32 for stability, then back to model dtype
                precond = torch.pow(pc.float(), p).to(dtype)
                logger.debug("OTA Task (%s): Using power p=%.3f", self.weight_info.name, p) # New logging

                precond_stats = precond.float()
                logger.debug(
                    "OTA Task (%s): Model %s - v^p curvature (p=%.3f) stats: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e" %
                    (self.weight_info.name, m, p, precond_stats.min(), precond_stats.max(), precond_stats.mean(), precond_stats.std())
                )

                # ---- Normalisation ablation ----
                scale_val: Optional[torch.Tensor] = None # For logging
                if norm_mode == "global":
                    scale = precond.float().mean().to(precond.dtype)
                    precond = precond / (scale + self.epsilon)
                    scale_val = scale
                elif norm_mode == "layer":
                    dims = list(range(1, precond.dim()))  # keep param axis (dim 0 is model index)
                    if not dims: # scalar tensor, treat as global
                        scale = precond.float().mean().to(precond.dtype)
                    else:
                        scale = precond.float().mean(dim=dims, keepdim=True).to(precond.dtype)
                    precond = precond / (scale + self.epsilon)
                    scale_val = scale
                elif norm_mode == "inv":
                    # For "inv" mode, precond (sqrt(pc)) is not scaled here.
                    # The inversion is applied when calculating weight_tensor.
                    pass # Will be handled later
                elif norm_mode == "inv-global":
                    # For "inv-global" mode, precond (sqrt(pc)) is not scaled here.
                    # The inversion and global scaling is applied when calculating weight_tensor.
                    pass # Will be handled later
                elif norm_mode != "none":
                    raise ValueError(f"Unknown normalisation mode for OTA: {norm_mode}")
                # ---------------------------------
                
                # Log post-normalisation statistics for 'precond'
                # For 'inv' and 'none', this logs stats of original sqrt(pc) or scaled if 'global'/'layer'.
                post_norm_stats = precond.float() 
                log_scale_val = "N/A"
                if scale_val is not None:
                    if scale_val.numel() == 1:
                        log_scale_val = f"{scale_val.item():.4e}"
                    else: # layer norm, scale is a tensor
                        log_scale_val = f"tensor(Min={scale_val.min().item():.4e}, Max={scale_val.max().item():.4e}, Mean={scale_val.mean().item():.4e})"
                
                logger.debug(
                    "OTA Task (%s): Model %s - Post-normalisation (mode: %s, scale: %s) precond stats: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
                    self.weight_info.name, m, norm_mode if norm_mode not in ["inv", "inv-global"] else "none (prior to inv/inv-global op)", log_scale_val, # Adjust log for inv modes
                    post_norm_stats.min(), post_norm_stats.max(), post_norm_stats.mean(), post_norm_stats.std()
                )

                if norm_mode == "inv-global":
                    inv_tensor = 1.0 / (precond + self.epsilon)
                    scale = inv_tensor.float().mean().to(inv_tensor.dtype) # Calculate mean in float32 for stability
                    weight_tensor = inv_tensor / (scale + self.epsilon)
                    logger.debug(
                        "OTA Task (%s): Model %s - inv-global: scale=%.4e. Stats after inv-global: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
                        self.weight_info.name, m, scale.item(), 
                        weight_tensor.float().min(), weight_tensor.float().max(), weight_tensor.float().mean(), weight_tensor.float().std()
                    )
                elif norm_mode == "inv":
                    # precond is sqrt(pc)
                    weight_tensor = 1.0 / (precond + self.epsilon)
                    if self.clip_factor is not None and self.clip_factor > 0:
                        cf = self.clip_factor
                        med = weight_tensor.median()
                        threshold = cf * med
                        before_max = weight_tensor.max()
                        weight_tensor = torch.clamp(weight_tensor, max=threshold)
                        # Check if any clipping actually occurred to avoid verbose logging if not
                        if before_max > threshold :
                            logger.debug(
                                "OTA Task (%s): Model %s - Clipped inv weights (factor %.1f*median=%.4e). "
                                "Max before %.4e -> after %.4e. Median was %.4e.",
                                self.weight_info.name, m, cf, threshold, before_max, weight_tensor.max(), med
                            )
                        else:
                            logger.debug(
                                "OTA Task (%s): Model %s - Inv weights (factor %.1f*median=%.4e) did not exceed clip threshold. "
                                "Max was %.4e. Median was %.4e.",
                                self.weight_info.name, m, cf, threshold, before_max, med
                            )
                else: # "none", "global", "layer"
                    # For "none", precond is sqrt(pc)
                    # For "global"/"layer", precond is scaled sqrt(pc)
                    weight_tensor = precond + self.epsilon
                                
                weight_tensor_stats = weight_tensor.float()
                logger.debug(
                    "OTA Task (%s): Model %s - Final weight_tensor (precond + eps) stats: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
                    self.weight_info.name, m, weight_tensor_stats.min(), weight_tensor_stats.max(), weight_tensor_stats.mean(), weight_tensor_stats.std()
                )
            else:
                # If a preconditioner path was provided but loading failed, emit a
                # reminder that we are falling back to the scalar weight.
                if self.tensor_parameters[m].get("preconditioner_path"):
                    logger.warning(
                        "OTA Task (%s): Model %s - Falling back to scalar weight because preconditioner could not be loaded (see previous warnings).",
                        self.weight_info.name,
                        m,
                    )
                else:
                    logger.info(
                        "OTA Task (%s): Model %s - No preconditioner_path, using scalar weight.",
                        self.weight_info.name,
                        m,
                    )
                # Fallback: scalar weight.
                scalar_w = self.tensor_parameters[m].get("weight", 1.0)
                logger.info(
                    "OTA Task (%s): Model %s - Using scalar weight: %s",
                    self.weight_info.name,
                    m,
                    scalar_w,
                )
                wt = torch.tensor(scalar_w, dtype=dtype, device=device)
                # broadcast to full shape
                while wt.dim() < stacked[i].dim():
                    wt = wt.unsqueeze(-1)
                weight_tensor = wt.expand_as(stacked[i]) + self.epsilon  # (w_i + ε)

            weight_tensors.append(weight_tensor)

        weights_stack = torch.stack(weight_tensors, dim=0)
        logger.debug("OTA Task (%s): weights_stack shape: %s", self.weight_info.name, weights_stack.shape)

        # Calculate element-wise std dev of weights across models
        if weights_stack.shape[0] > 1: 
            elementwise_std_of_weights = torch.std(weights_stack, dim=0, unbiased=True)
            elementwise_std_stats = elementwise_std_of_weights.float()
            logger.info(
                "OTA Task (%s): Element-wise StdDev of final weights across models: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
                self.weight_info.name, 
                elementwise_std_stats.min(), 
                elementwise_std_stats.max(), 
                elementwise_std_stats.mean(), 
                elementwise_std_stats.std()
            )

            # --- New section for saving detailed tensor data ---
            TENSORS_FOR_DETAILED_ANALYSIS = [
                # --- High Mean Element-wise StdDev (High Average Disagreement) ---
                "model.layers.0.post_attention_layernorm.weight", 
                "model.layers.0.input_layernorm.weight",          
                "model.layers.1.post_attention_layernorm.weight", 
                "model.layers.5.post_attention_layernorm.weight",
                "model.layers.10.post_attention_layernorm.weight",
                "model.layers.15.post_attention_layernorm.weight",
                "model.layers.20.post_attention_layernorm.weight",
                "model.layers.25.post_attention_layernorm.weight",
                "model.layers.31.post_attention_layernorm.weight",
                "model.layers.0.self_attn.v_proj.weight",         
                "model.layers.5.self_attn.v_proj.weight",
                "model.layers.10.self_attn.v_proj.weight",
                "model.layers.15.self_attn.v_proj.weight",
                "model.layers.20.self_attn.v_proj.weight",
                "model.layers.25.self_attn.v_proj.weight",

                # --- Low Mean Element-wise StdDev (High Average Agreement) ---
                "model.layers.0.self_attn.q_proj.weight",         
                "model.layers.0.self_attn.k_proj.weight",         
                "model.layers.10.self_attn.q_proj.weight",
                "model.layers.20.self_attn.k_proj.weight",
                "model.layers.23.self_attn.q_proj.weight",        

                # --- High Max Element-wise StdDev (High Peak Disagreement) ---
                # model.layers.0.self_attn.v_proj.weight is already included above
                "lm_head.weight",                                 
                "model.embed_tokens.weight",                      
                "model.layers.2.self_attn.v_proj.weight",         
                "model.layers.12.self_attn.v_proj.weight",        
                "model.layers.22.self_attn.v_proj.weight",        
                "model.layers.1.mlp.down_proj.weight",            
                "model.layers.6.mlp.down_proj.weight",            
                "model.layers.11.mlp.down_proj.weight",           

                # --- Other Representative Types ---
                "model.layers.0.mlp.gate_proj.weight",            
                "model.layers.15.mlp.gate_proj.weight",           
                "model.layers.30.mlp.gate_proj.weight",           
                "model.norm.weight"                               
            ]
            
            detailed_log_base_dir = "." # Default to current directory
            if self.out_path_for_detailed_logs:
                detailed_log_base_dir = self.out_path_for_detailed_logs

            DETAILED_LOG_OUTPUT_DIR = os.path.join(detailed_log_base_dir, "ota_detailed_tensor_logs")

            if self.weight_info.name in TENSORS_FOR_DETAILED_ANALYSIS:
                try:
                    if not os.path.exists(DETAILED_LOG_OUTPUT_DIR):
                        os.makedirs(DETAILED_LOG_OUTPUT_DIR, exist_ok=True)
                    
                    model_identifiers = []
                    for m_ref in models: # models is List[ModelReference] from earlier in execute()
                        actual_model_path_or_id = None
                        # m_ref.model can be a str or a ModelDefinition object
                        if hasattr(m_ref.model, 'path') and m_ref.model.path: # ModelDefinition with path
                            actual_model_path_or_id = m_ref.model.path
                        elif hasattr(m_ref.model, 'repo_id') and m_ref.model.repo_id: # ModelDefinition with repo_id
                            actual_model_path_or_id = m_ref.model.repo_id
                        elif isinstance(m_ref.model, str): # Direct string path
                            actual_model_path_or_id = m_ref.model
                        
                        if actual_model_path_or_id:
                            # Heuristic: if it's a checkpoint path, get the parent dir name
                            potential_checkpoint_name = os.path.basename(actual_model_path_or_id)
                            parent_dir_path = os.path.dirname(actual_model_path_or_id)
                            parent_dir_name = os.path.basename(parent_dir_path)

                            if potential_checkpoint_name.startswith("checkpoint-") and parent_dir_name:
                                model_name_for_manifest = parent_dir_name
                            elif potential_checkpoint_name: # Not a checkpoint path, or no parent_dir_name if top-level
                                model_name_for_manifest = potential_checkpoint_name
                            elif parent_dir_name: # Path ended with /, checkpoint name was empty
                                 model_name_for_manifest = parent_dir_name
                            else: # Fallback to the full string if all else fails
                                model_name_for_manifest = actual_model_path_or_id
                            model_identifiers.append(model_name_for_manifest)
                        else:
                            # Fallback for ModelDefinition objects without path or repo_id (should be rare)
                            # Or if m_ref.model was some other unexpected type
                            model_identifiers.append(f"unknown_model_{len(model_identifiers)}")
                    
                    manifest_path = os.path.join(DETAILED_LOG_OUTPUT_DIR, "model_manifest.json")
                    if not os.path.exists(manifest_path):
                        with open(manifest_path, 'w') as f_manifest:
                            json.dump(model_identifiers, f_manifest, indent=2)
                        logger.info("OTA Task (%s): Saved model manifest to %s (Models: %s)", self.weight_info.name, manifest_path, model_identifiers)

                    sanitized_tensor_name = self.weight_info.name.replace("/", "_").replace(".", "-")
                    
                    weights_stack_path = os.path.join(DETAILED_LOG_OUTPUT_DIR, f"{sanitized_tensor_name}_weights_stack.pt")
                    torch.save(weights_stack.cpu(), weights_stack_path)
                    logger.info("OTA Task (%s): Saved detailed weights_stack to %s", self.weight_info.name, weights_stack_path)
                    
                    ew_std_path = os.path.join(DETAILED_LOG_OUTPUT_DIR, f"{sanitized_tensor_name}_elementwise_std.pt")
                    torch.save(elementwise_std_of_weights.cpu(), ew_std_path)
                    logger.info("OTA Task (%s): Saved detailed elementwise_std_of_weights to %s", self.weight_info.name, ew_std_path)

                except Exception as e:
                    logger.error("OTA Task (%s): Failed to save detailed tensor data for %s. Error: %s", self.weight_info.name, self.weight_info.name, e)
            # --- End of new section ---
        else:
            logger.info(
                "OTA Task (%s): Only one model contributing or fallback for all, skipping element-wise StdDev of weights across models.",
                self.weight_info.name
            )

        # Compute OTA merge: element‑wise weighted average.
        numerator = (weights_stack * stacked).sum(dim=0)
        # Denominator already contains  Σ (P_i + ε)  ⇒  Σ P_i + n·ε
        denominator = weights_stack.sum(dim=0)
        merged = numerator / denominator
        merged_stats = merged.float()
        logger.info(
            "OTA Task (%s): Final merged tensor stats: Min=%.4e, Max=%.4e, Mean=%.4e, Std=%.4e",
            self.weight_info.name, merged_stats.min(), merged_stats.max(), merged_stats.mean(), merged_stats.std()
        )
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
        return [
            ConfigParameterDef(name="epsilon", required=False, default_value=1e-8),
            ConfigParameterDef(name="normalise", required=False, default_value="none", description="How to normalize preconditioner tensors before use: 'none', 'global', 'layer', 'inv', 'inv-global'"),
            ConfigParameterDef(name="clip_factor", required=False, default_value=None, description="For \"inv\" mode, clips weights at N * median (e.g., 5.0). Set to None or null to disable clipping. Ignored if normalise is 'inv-global'."),
            ConfigParameterDef(
                name="power",
                required=False,
                default_value=0.5,
                description="Exponent p used for curvature weighting: weight ∝ v^p. Typical values: 0.25, 0.5, 1.0. Must be > 0."
            )
        ]

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
        base_model: Optional[ModelReference],
        merge_options: Optional[MergeOptions] = None,
        out_path_for_debug: Optional[str] = None
    ) -> Task:
        if out_path_for_debug:
            logger.debug("OTAMerge.make_task: out_path_for_debug is set to: %s", out_path_for_debug)
        else:
            logger.debug("OTAMerge.make_task: out_path_for_debug was not provided.")

        task = OTAMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            epsilon=parameters.get("epsilon", 1e-8),
            normalise_mode=str(parameters.get("normalise", "none")).lower(),
            weight_info=output_weight,
            merge_options=merge_options,
            out_path_for_detailed_logs=out_path_for_debug,
            clip_factor=parameters.get("clip_factor"),
            precond_power=parameters.get("power", 0.5)
        )
        return task 