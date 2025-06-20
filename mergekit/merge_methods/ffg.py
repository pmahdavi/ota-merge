from typing import Any, Dict, List, Optional

import logging

import torch
from typing_extensions import override
from pydantic import ConfigDict

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

# We reuse the helper functions that OTA already provides for loading Adam
# second-moment (exp_avg_sq) tensors from either *.safetensors files or torch
# optimizer checkpoints.  Those helpers are cached, so importing them here is
# cheap and avoids duplicated file-handling logic.
from mergekit.merge_methods.ota import _open_safetensors, _open_torch_opt_state  # type: ignore

logger = logging.getLogger(__name__)


class FFGTask(Task[torch.Tensor]):
    """Graph task that performs Fast-Fisher-Graft for a single tensor."""

    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    density: float  # proportion of elements to *keep* from the tuned delta
    metric: str  # "fisher" | "magnitude"
    base_model: ModelReference  # identifies w₀ (pre-train)
    weight_info: WeightInfo

    # Pydantic v2: make subclass immutable (same as parent Task)
    model_config = ConfigDict(frozen=True)

    # ---------------------------------------------------------------------
    # Task boiler-plate
    # ---------------------------------------------------------------------

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    # ---------------------------------------------------------------------
    # Helper: optionally load Adam v (exp_avg_sq) for the tuned model.
    # ---------------------------------------------------------------------

    def _load_fisher(self, model: ModelReference, tensor_shape: torch.Size) -> Optional[torch.Tensor]:
        """Attempt to load the `exp_avg_sq` tensor for *this* weight.

        Returns None if unavailable or shape mismatch.
        """
        params = self.tensor_parameters[model]
        path = params.get("preconditioner_path")
        if not path:
            return None

        logger.debug("FFG: Loading preconditioner for model %s from %s", model, path)

        is_safetensors = path.lower().endswith(".safetensors")
        try:
            st = _open_safetensors(path) if is_safetensors else _open_torch_opt_state(path)
        except Exception as e:
            logger.warning("FFG: Failed to open preconditioner file '%s': %s", path, e)
            return None

        # Try exact key, else common fall-backs.
        key = self.weight_info.name
        if key not in st.keys():  # type: ignore[attr-defined]
            fallbacks = [f"{key}.exp_avg_sq", key.replace("weight", "weight.exp_avg_sq")]
            key = next((k for k in fallbacks if k in st.keys()), None)  # type: ignore[attr-defined]
            if key is None:
                logger.warning(
                    "FFG: exp_avg_sq tensor for %s not found in %s", self.weight_info.name, path
                )
                return None
        try:
            pc = st.get_tensor(key) if is_safetensors else st[key]  # type: ignore[index]
        except Exception as e:
            logger.warning("FFG: Error reading tensor '%s' from %s: %s", key, path, e)
            return None

        if pc.shape != tensor_shape:
            logger.warning(
                "FFG: Shape mismatch for preconditioner of %s (got %s, expected %s)",
                self.weight_info.name,
                pc.shape,
                tensor_shape,
            )
            return None
        return pc

    # ---------------------------------------------------------------------
    # Core execution
    # ---------------------------------------------------------------------

    def execute(self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs) -> torch.Tensor:  # noqa: N802
        if len(tensors) != 2:
            raise RuntimeError("FFG merge expects exactly two models (base + tuned)")
        if self.base_model not in tensors:
            raise RuntimeError("Base model not present in input tensors for FFG merge")

        # Identify base (w0) and tuned (wT) tensors
        base_tensor = tensors[self.base_model]
        tuned_model = next(m for m in tensors if m != self.base_model)
        tuned_tensor = tensors[tuned_model]

        # Align embeddings if needed
        rectify_embed_sizes(self.weight_info, [base_tensor, tuned_tensor])

        if base_tensor.shape != tuned_tensor.shape:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}: {base_tensor.shape} vs {tuned_tensor.shape}"
            )

        # Handle trivial density cases
        density_raw = self.density
        if density_raw <= 0:
            return base_tensor
        if density_raw >= 1:
            return tuned_tensor

        # Calculate delta
        delta = (tuned_tensor - base_tensor).to(tuned_tensor.dtype)

        # Build importance scores
        fisher_tensor: Optional[torch.Tensor] = None
        if self.metric == "fisher":
            fisher_tensor = self._load_fisher(tuned_model, tuned_tensor.shape)
            if fisher_tensor is None:
                logger.warning(
                    "FFG: metric='fisher' requested but no preconditioner available – falling back to |delta|"
                )

        if fisher_tensor is not None:
            fisher_tensor = fisher_tensor.to(device=delta.device, dtype=delta.dtype)
            scores = delta.pow(2) * fisher_tensor
        else:
            scores = delta.abs()

        # Top-k mask
        k = int(max(1, round(density_raw * scores.numel())))
        # torch.topk is faster but we need indices
        flat_scores = scores.flatten()
        _, topk_idx = torch.topk(flat_scores, k=k, largest=True, sorted=False)
        mask = torch.zeros_like(flat_scores, dtype=delta.dtype)
        mask[topk_idx] = 1.0
        mask = mask.view_as(delta)

        grafted = base_tensor + delta * mask
        return grafted

    # ---------------- Info helpers ----------------

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class FFGMerge(MergeMethod):
    """Fast Fisher Graft merge method."""

    def name(self) -> str:  # noqa: D401 N802
        return "ffg"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Fast Fisher Graft"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2406.11617"  # Placeholder paper link

    # ---------------- Parameter definitions ----------------

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", required=False, default_value=0.1),
            ConfigParameterDef(name="metric", required=False, default_value="fisher"),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        # per-model optional field for fisher path
        return [ConfigParameterDef(name="preconditioner_path", required=False)]

    # ---------------- Task factory ----------------

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        if base_model is None:
            raise RuntimeError("ffg merge requires 'base_model' to be set in config")
        density_raw = parameters.get("density", 0.1)
        if density_raw is None:
            density = 0.0
        else:
            density = float(density_raw)
        metric = str(parameters.get("metric", "fisher")).lower()
        if metric not in ("fisher", "magnitude"):
            raise ValueError("ffg parameter 'metric' must be 'fisher' or 'magnitude'")

        return FFGTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            density=density,
            metric=metric,
            base_model=base_model,
            weight_info=output_weight,
        ) 