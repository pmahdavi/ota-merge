# OTA-Merge: Curvature-Informed Model Merging

This repository implements **OTA (Optimization Trajectory Aware) Merging**, a novel model merging method that leverages optimization dynamics for curvature-informed merging of fine-tuned language models. Built on top of [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit), OTA introduces Fast Fisher Grafting (FFG) and curvature-aware aggregation using optimizer second-moment statistics.

> **Paper:** [Harnessing Optimization Dynamics for Curvature-Informed Model Merging](https://arxiv.org/abs/2509.11167)  
> **Visualization Tools:** https://github.com/pmahdavi/surgeon.git

## Key Features

- **Curvature-Aware Merging**: Uses Adam's second moments as Fisher information proxy for OTA
- **Fast Fisher Grafting**: Task localization and interference mitigation (see [surgeon repo](https://github.com/pmahdavi/surgeon.git))
- **Memory Efficiency**: Rank-1 approximation for second-moment compression
- **Multiple Methods**: OTA, Linear, TIES, DARE, and Breadcrumbs implementations
- **Production Ready**: Local execution and PBS cluster support

## Artifacts

### Available Checkpoints

All SFT models are fine-tuned from Meta-Llama-3.1-8B and available on HuggingFace:

| Capability | Model ID | Preconditioner Path |
|------------|----------|---------------------|
| **Math Reasoning** | `pmahdavi/Llama-3.1-8B-math-reasoning` | `/export/exp_avg_sq.safetensors` |
| **Coding** | `pmahdavi/Llama-3.1-8B-coding` | `/export/exp_avg_sq.safetensors` |
| **Coding 2** | `pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4` | `/export_full_state_checkpoint-1100/exp_avg_sq.safetensors` |
| **Precise IF** | `pmahdavi/Llama-3.1-8B-precise-if` | `/export/exp_avg_sq.safetensors` |
| **General** | `pmahdavi/Llama-3.1-8B-general` | `/export/exp_avg_sq.safetensors` |
| **Knowledge Recall** | `pmahdavi/Llama-3.1-8B-knowledge-recall` | `/export/exp_avg_sq.safetensors` |

Access models at: https://huggingface.co/pmahdavi
### Release Artifacts

The [v1.0.0-rc.1 release](https://github.com/pmahdavi/ota-merge/releases/tag/v1.0.0-rc.1) includes the following analysis PDFs:

- **[`FFG_SFT_Localization.pdf`](https://github.com/pmahdavi/ota-merge/releases/download/v1.0.0-rc.1/FFG_SFT_Localization.pdf)** - Visualization of Fast Fisher Grafting task localization across different SFT models.
- **[`all_layers_weights_curvature_grid_math_vs_code.pdf`](https://github.com/pmahdavi/ota-merge/releases/download/v1.0.0-rc.1/all_layers_weights_curvature_grid_math_vs_code.pdf)** - Layer-wise curvature comparison of math and coding SFT checkpoints.
- **[`all_layers_weights_curvature_grid_code_vs_code2.pdf`](https://github.com/pmahdavi/ota-merge/releases/download/v1.0.0-rc.1/all_layers_weights_curvature_grid_code_vs_code2.pdf)** - Layer-wise curvature comparison between two coding SFT models (cosine vs. WSD schedules).
- - **[`all_layers_curv_FFG40dense.pdf`](https://github.com/pmahdavi/ota-merge/releases/download/v1.0.0-rc.1/all_layers_curv_FFG40dense.pdf)** - Combined Layer-wise curvature comparison with 40% density FFG masking.



## Installation

```bash
# Clone the repository
git clone https://github.com/pmahdavi/ota-merge.git
cd ota-merge

# Install in development mode
pip install -e .
```

## Quick Start

### Running Merges Locally

The primary entry point is `run_merge.py`, which provides a convenient wrapper around mergekit:

```bash
# Run OTA+FFG merge locally
python run_merge.py configs/hf_ota_ffg_per_model_thresh.yml --local

# Run with custom output directory
python run_merge.py configs/hf_ota_ffg_per_model_thresh.yml \
    --output_dir ./my-merged-models --local
```

### PBS Cluster Execution

For Penn State cluster users or similar PBS systems:

```bash
# Submit to PBS queue
python run_merge.py configs/hf_ota_ffg_per_model_thresh.yml

# Customize PBS resources
python run_merge.py configs/hf_ota_ffg_per_model_thresh.yml \
    --walltime 24:00:00 --ngpus 4 --mem 120g
```

## Configuration Files

All configurations are in the `configs/` directory. Key configurations used in our paper:

| Method | Config File | Description |
|--------|-------------|-------------|
| **OTA+FFG** | `hf_ota_ffg_per_model_thresh.yml` | Main OTA results with per-model thresholds |
| **OTA+FFG (Rank-1)** | `hf_ota_ffg_rank1lemma.yml` | Memory-efficient rank-1 approximation |
| **OTA (w/ Linear)** | Enable `masked_task_vector_merge_method: linear` | Linear merge after FFG masking |
| **OTA (w/o FFG)** | `hf_ota.yml` | Pure OTA without FFG denoising |
| **DARE-TIES** | `latest_llama_factory_dare.yml` | DARE with TIES sign consensus |
| **Linear** | `latest_llama_factory_linear.yml` | Simple weight averaging |
| **TIES** | `latest_llama_factory_ties.yml` | TIES merging baseline |
| **Breadcrumbs-TIES** | `latest_llama_factory_breadcrumbs_ties.yml` | Model breadcrumbs with TIES |

## Configuration Guide

### Understanding YAML Configurations

All merge configurations follow a standard YAML format with these key sections:
- `merge_method`: The merging algorithm to use (e.g., `ota`, `linear`, `ties`, `dare_ties`)
- `base_model`: Reference model for task arithmetic methods (required for some methods)
- `parameters`: Global parameters that apply to all models
- `models`: List of models to merge with their individual parameters
- `dtype`: Data type for computation (typically `bfloat16` or `float16`)
- `tokenizer_source`: How to handle tokenizer (`union` combines all vocabularies)

### OTA-Specific Configuration

OTA introduces several unique parameters for curvature-aware merging:

```yaml
merge_method: ota
base_model: meta-llama/Meta-Llama-3.1-8B

parameters:
  # Global OTA parameters
  epsilon: 1e-26                      # Numerical stability for preconditioner
  normalise: none                     # Options: none, mean, sum
  power: 0.5                         # Preconditioner power (0.5 = sqrt, 1.0 = linear)
  fallback_to_base: true             # Use base weights when parameters are masked
  rescale: false                     # Rescale task vectors after masking
  approximate_norm: false            # Use approximate rescaling (faster)
  rescale_relative_threshold: 0.1    # Threshold for rescaling (if rescale=true)
  
  # Optional: Post-FFG merge strategy
  # masked_task_vector_merge_method: linear  # Options: precond, linear, ties

models:
  - model: pmahdavi/Llama-3.1-8B-math-reasoning
    parameters:
      # REQUIRED: Path to optimizer second moments
      preconditioner_path: pmahdavi/Llama-3.1-8B-math-reasoning/export/exp_avg_sq.safetensors
      
      # FFG threshold - determines masking aggressiveness
      precond_threshold: 2.626e-19     # Lower = more parameters kept
      
      # Optional: Memory-efficient rank-1 approximation
      rank1_approx: true              # Uses AdaFactor-style compression
      # rank: 16                      # Alternative: specify rank directly
      
  - model: pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4
    parameters:
      preconditioner_path: pmahdavi/Llama-3.1-8B-coding-tulu3-ebs128-lr5e6-wsdcr0p4/export_full_state_checkpoint-1100/exp_avg_sq.safetensors
      precond_threshold: 1.1152e-18

dtype: bfloat16
tokenizer_source: union
```

#### Key OTA Parameters Explained:

1. **`preconditioner_path`** (required): Path to the Adam optimizer's second moments (`exp_avg_sq.safetensors`). Can be:
   - HuggingFace format: `model_id/path/to/file` or `model_id:path/to/file`
   - Local path: `/absolute/path/to/exp_avg_sq.safetensors`

2. **`precond_threshold`**: Controls FFG sparsity. Parameters with saliency below this threshold are masked.
   - Lower values = more parameters retained
   - Can be global (in `parameters`) or per-model

3. **`power`**: Exponent applied to preconditioner values
   - `0.5` (default): Square root scaling, standard for Adam-based preconditioning
   - `1.0`: Linear scaling, uses raw second moments

4. **`rank1_approx`**: Memory-efficient compression using AdaFactor-style approximation
   - Reduces storage from O(mn) to O(m+n) per parameter matrix

5. **`fallback_to_base`**: Use base model values for masked parameters (default: true)

6. **`masked_task_vector_merge_method`**: Post-FFG merge strategy. Options:
   - `precond` (default): Merges masked task vectors using a weighted average based on the preconditioner values.
   - `linear`: Performs a simple, unweighted average of the masked task vectors.
   - `ties`: Applies the TIES-merging algorithm (trim, elect, disentangle) to the masked task vectors.

### Baseline Method Configurations

For other merging methods, refer to the [original mergekit documentation](https://github.com/arcee-ai/mergekit#merge-methods). Here are quick examples:

**Linear (Simple Averaging):**
```yaml
merge_method: linear
models:
  - model: model1
    parameters:
      weight: 1.0  # Equal weights
  - model: model2
    parameters:
      weight: 1.0
```

**TIES:**
```yaml
merge_method: ties
base_model: meta-llama/Meta-Llama-3.1-8B
parameters:
  density: 0.5    # Fraction of parameters to keep
  weight: 1.0     # Model weighting
models:
  - model: model1
    parameters:
      density: 0.45  # Can override per-model
```

**DARE:**
```yaml
merge_method: dare_ties  # or dare_linear
base_model: meta-llama/Meta-Llama-3.1-8B
parameters:
  density: 0.95   # High density = less pruning
```

### Configuration Tips

1. **Finding Optimal Thresholds**: Run FFG at various densities, evaluate the grafted models, and choose the smallest density that maintains expert performance. Extract the threshold from `statistics.json` in the FFG output directory (see surgeon repo for details).

2. **Memory Optimization**: Enable `rank1_approx: true` for large models to reduce memory footprint.

3. **Debugging**: Use `--dry-run` to preview the full command before execution.

4. **Path Formats**: Preconditioner paths can be HuggingFace (`model_id:path`) or absolute local paths.

## How `run_merge.py` Works

The `run_merge.py` script is a sophisticated wrapper that:

1. **Parses YAML configs** to extract merge method and model names
2. **Generates adaptive output names** based on:
   - Merge method (e.g., `ota-ffg`, `dare_ties`)
   - Model combinations (shortened names like `math-reasoning_coding`)
   - Hyperparameters (e.g., `d0.5` for density, `precond-thresh2.6e-19`)
3. **Handles both local and PBS execution**:
   - Local: Runs mergekit directly with conda environment
   - PBS: Generates optimized job scripts with proper resource allocation
4. **Manages GPU memory** with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
5. **Supports dry runs** to preview commands before execution

Example generated output directory:
```
ota-ffg_math-reasoning_coding_precise-if_general_knowledge-recall_precond-thresh-multi/
```

## Method Overview

OTA leverages Adam's second-moment estimates (`exp_avg_sq`) as a diagonal Fisher information proxy. The method operates in two stages:

1. **Fast Fisher Grafting (FFG)**: Masks low-saliency parameter updates based on saliency score `s_i = H_ii * (Δw_i)²`
2. **Curvature-Aware Aggregation**: Merges denoised models using preconditioner weighting


## Reproducing Paper Results

To reproduce the results from our paper:

```bash
# Main OTA+FFG results
python run_merge.py configs/hf_ota_ffg_per_model_thresh.yml --local

# Rank-1 approximation comparison
python run_merge.py configs/hf_ota_ffg_rank1lemma.yml --local

# Baseline comparisons
python run_merge.py configs/latest_llama_factory_linear.yml --local
python run_merge.py configs/latest_llama_factory_ties.yml --local
python run_merge.py configs/latest_llama_factory_dare.yml --local
python run_merge.py configs/latest_llama_factory_breadcrumbs_ties.yml --local
```

## Visualizations and Analysis

For detailed FFG implementation, task vector localization visualizations, and curvature analysis, see the surgeon repository: https://github.com/pmahdavi/surgeon.git

The surgeon repo provides:
- Complete FFG implementation and grafting experiments
- Task vector density and localization visualizations
- Curvature heatmap generation
- Mask overlap and sparsity distribution analysis
- Layer-wise and parameter-type breakdowns

## Evaluation

We use [OLMES (Open Language Model Evaluation System)](https://github.com/pmahdavi/olmes) for evaluating merged models. The evaluation suite includes all benchmarks reported in our paper. 

To evaluate your merged models:
1. Use the merged models produced by this repository or the checkpoints listed in the [Artifacts](#artifacts) section
2. Follow the evaluation instructions in our OLMES fork


## Citation

If you use OTA-Merge in your research, please cite:

```bibtex
@misc{mahdavinia2025harnessingoptimizationdynamicscurvatureinformed,
      title={Harnessing Optimization Dynamics for Curvature-Informed Model Merging}, 
      author={Pouria Mahdavinia and Hamed Mahdavi and Niloofar Mireshghallah and Mehrdad Mahdavi},
      year={2025},
      eprint={2509.11167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.11167}, 
}
```

For the underlying mergekit framework:

```bibtex
@inproceedings{goddard-etal-2024-arcees,
    title = "Arcee{'}s {M}erge{K}it: A Toolkit for Merging Large Language Models",
    author = "Goddard, Charles  and
      Siriwardhana, Shamane  and
      Ehghaghi, Malikeh  and
      Meyers, Luke  and
      Karpukhin, Vladimir  and
      Benedict, Brian  and
      McQuade, Mark  and
      Solawetz, Jacob",
    editor = "Dernoncourt, Franck  and
      Preo{\c{t}}iuc-Pietro, Daniel  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-industry.36",
    doi = "10.18653/v1/2024.emnlp-industry.36",
    pages = "477--485",
}
```

## License

This project builds upon mergekit and is released under the same license. See LICENSE for details.
