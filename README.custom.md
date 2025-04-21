# MergeKit Reference Guide

## Introduction

MergeKit is a powerful toolkit for merging pre-trained language models. It uses an out-of-core approach to perform elaborate merges in resource-constrained situations. Merges can be run entirely on CPU or accelerated with as little as 8 GB of VRAM. Many merging algorithms are supported, making it a versatile tool for LLM research and development.

## Installation

```bash
# Clone the repository
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

# Create and activate conda environment
conda create -n mergekit python=3.10 -y
conda activate mergekit

# Install the package
pip install -e .
```

## Key Features

- **Resource Efficiency**: Performs merges with minimal resources through out-of-core processing
- **Multiple Backends**: Runs on CPU or GPU with minimal VRAM requirements
- **Wide Architecture Support**: Works with Llama, Mistral, GPT-NeoX, StableLM, Mixtral, and more
- **Diverse Merge Methods**: Implements many merging algorithms (TIES, DARE, Task Arithmetic, etc.)
- **Flexible Configuration**: Uses YAML-based configuration for easy customization
- **Advanced Tools**: Includes LoRA extraction, MoE merging, and evolutionary optimization

## Architecture and Workflow

MergeKit operates through a well-structured pipeline that handles model merging efficiently:

### Core Components

1. **Configuration System**: Defines merge parameters through YAML files
2. **Model Loading**: Efficiently loads model weights using a lazy-loading cache
3. **Merge Planning**: Creates a computational plan for merging tensors
4. **Task Execution**: Applies merge algorithms to tensors according to the plan
5. **Model Saving**: Handles saving the merged model with proper tokenizer and configuration

### Job Submission System (`run_merge.py`)

The `run_merge.py` script is a job management utility that:

1. **Parses a merge configuration** from a YAML file
2. **Generates an adaptive output directory name** based on the merge method and models
3. **Creates a PBS job script** for running on a cluster
4. **Submits the job** and provides tracking information

Example usage:
```bash
# Submit a merging job
python run_merge.py examples/linear.yml --ngpus 2 --ncpus 16 --mem 80g --walltime 12:00:00
```

### Execution Flow

1. **Configuration Parsing**: The YAML config is validated and normalized
2. **Architecture Analysis**: Model architectures are analyzed to ensure compatibility
3. **Merge Planning**: A directed acyclic graph of operations is created
4. **Task Execution**: Tasks are executed in dependency order
5. **Output Generation**: The merged model is saved with updated configuration

## Supported Merge Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| **Linear** | Simple weighted average | `weight`, `normalize` |
| **SLERP** | Spherical interpolation | `t` (interpolation factor) |
| **NuSLERP** | Enhanced spherical interpolation | `weight`, `nuslerp_flatten`, `nuslerp_row_wise` |
| **Task Arithmetic** | Model differences as task vectors | `weight`, `lambda` (scaling factor) |
| **TIES** | Task vectors with sign consensus | `weight`, `density` |
| **DARE** | Random pruning with rescaling | `weight`, `density` |
| **Model Breadcrumbs** | Discards small and large differences | `weight`, `density`, `gamma` |
| **Model Stock** | Geometric weight optimization | `filter_wise` |
| **DELLA** | Adaptive magnitude-based pruning | `weight`, `density`, `epsilon` |
| **Passthrough** | No-op for layer stacking | None |

## Implementation Details

### Linear
- **Implementation**: Computes a weighted average of corresponding tensors: `result = Σ(weight_i * tensor_i) / Σ(weight_i)`
- **Normalization**: When `normalize=True`, weights are divided by their sum to ensure they sum to 1
- **Complexity**: O(n) where n is the number of parameters in the models
- **Code Path**: Implemented in `mergekit/merge_methods/linear.py`

### SLERP (Spherical Linear Interpolation)
- **Implementation**: Treats tensors as points on a hypersphere and interpolates along the geodesic
- **Formula**: `result = sin((1-t)*θ)/sin(θ) * tensor1 + sin(t*θ)/sin(θ) * tensor2` where θ is the angle between tensors
- **Limitation**: Works with exactly two models, with one being the base model
- **Code Path**: Implemented in `mergekit/merge_methods/slerp.py`

### NuSLERP
- **Implementation**: Enhanced version of SLERP with more options
- **Row-wise Mode**: Can spherically interpolate individual row/column vectors instead of whole tensors
- **Task Vector Compatibility**: Can operate on task vectors (differences from base model) rather than raw weights
- **Code Path**: Implemented in `mergekit/merge_methods/nuslerp.py`

### Task Arithmetic
- **Implementation**: Calculates task vectors by subtracting base model from each fine-tuned model
- **Combination**: Forms weighted sum of task vectors and adds back to base model
- **Scaling**: Applies `lambda` scaling factor to final task vector before adding to base model
- **Formula**: `result = base + λ * Σ(weight_i * (model_i - base))`
- **Code Path**: Implemented in `mergekit/merge_methods/generalized_task_arithmetic.py`

### TIES
- **Implementation**: Extends task arithmetic with interference elimination
- **Sparsification**: Keeps only top-k% magnitude elements in each task vector based on `density` parameter
- **Sign Consensus**: Elements retained only if their signs agree across most task vectors
- **Process**:
  1. Calculate task vectors: `task_i = model_i - base`
  2. Sparsify each task vector by keeping only top magnitude elements
  3. Apply sign consensus algorithm to eliminate interference
  4. Combine weighted sparse vectors and add to base model
- **Code Path**: Implemented as a variant in `mergekit/merge_methods/generalized_task_arithmetic.py`

### DARE (Dense-And-Randomly-Equivalent)
- **Implementation**: Uses random pruning instead of magnitude-based pruning in TIES
- **Probability Scaling**: Probability of keeping element proportional to its magnitude
- **Rescaling**: After pruning, rescales remaining values to preserve expected magnitude
- **Variants**: Available with (`dare_ties`) or without (`dare_linear`) sign consensus
- **Code Path**: Implemented as a variant in `mergekit/merge_methods/generalized_task_arithmetic.py`

### Model Breadcrumbs
- **Implementation**: Two-sided pruning approach that removes both small and extremely large differences
- **Parameters**: `density` controls overall sparsity; `gamma` determines percentage of largest differences to discard
- **Purpose**: Avoids both noise (small diffs) and outliers (extremely large diffs)
- **Variants**: Available with (`breadcrumbs_ties`) or without (`breadcrumbs`) sign consensus
- **Code Path**: Implemented as a variant in `mergekit/merge_methods/generalized_task_arithmetic.py`

### Model Stock
- **Implementation**: Uses geometric properties of fine-tuned models to compute optimal merge weights
- **Key Insight**: Weight for each model is proportional to cosine similarity between its task vector and average of all task vectors
- **Row-wise Mode**: When `filter_wise=True`, computes separate weights for each row/filter
- **Code Path**: Implemented in `mergekit/merge_methods/model_stock.py`

### DELLA (Density Elaborated with Linear Layer-wise Adaptation)
- **Implementation**: Adaptive magnitude-based pruning approach
- **Process**:
  1. Ranks parameters in each row of task vectors based on magnitude
  2. Assigns drop probabilities inversely proportional to rank (controlled by `epsilon`)
  3. Performs probabilistic pruning and rescales remaining parameters
- **Variants**: Available with (`della`) or without (`della_linear`) sign consensus
- **Code Path**: Implemented as a variant in `mergekit/merge_methods/generalized_task_arithmetic.py`

### Passthrough
- **Implementation**: Simply passes input tensors through unchanged
- **Purpose**: Acts as a no-op primarily used for layer stacking merges
- **Note**: Does not change values; only helps manage tensor flow in complex merges
- **Code Path**: Implemented in `mergekit/merge_methods/passthrough.py`

## Basic Usage

### 1. Create a YAML Configuration

```yaml
# linear_merge.yml
models:
  - model: model1/path
    parameters:
      weight: 0.6
  - model: model2/path
    parameters:
      weight: 0.4
merge_method: linear
dtype: float16
```

### 2. Run the Merge

**Direct Method**:
```bash
mergekit-yaml linear_merge.yml --out_dir ./merged_model
```

**Using Job Submission Script**:
```bash
python run_merge.py linear_merge.yml --ngpus 2 --ncpus 16
```

The `run_merge.py` script will:
1. Generate an appropriate merge name based on configuration
2. Create a PBS job script for cluster execution
3. Submit the job to the PBS scheduler
4. Provide job tracking information

## Advanced Usage

### Layer Merging

```yaml
# layer_merge.yml
slices:
  - sources:
    - model: model1/path
      layer_range: [0, 12]
    - model: model2/path
      layer_range: [12, 24]
merge_method: passthrough
```

### Parameter Control

```yaml
# task_arithmetic.yml
models:
  - model: base_model/path
  - model: finetuned1/path
    parameters:
      weight: 0.7
  - model: finetuned2/path
    parameters:
      weight: 0.3
merge_method: task_arithmetic
base_model: base_model/path
parameters:
  lambda: 0.5
```

### Tokenizer Configuration

```yaml
# with_tokenizer.yml
models:
  # Model definitions...
merge_method: linear
tokenizer:
  source: "union"  # Options: "union", "base", or specific model
  tokens:
    # Optional token embedding handling
    <|im_start|>:
      source: "chatml_model"
      force: true
chat_template: "llama-3"  # or custom template
```

## Memory Optimization

For low-memory environments:

```bash
# CPU-only
mergekit-yaml config.yml --out_dir ./merged_model --cpu-only

# Low VRAM usage (8GB+)
mergekit-yaml config.yml --out_dir ./merged_model --low-cpu-memory
```

## Advanced Tools

### LoRA Extraction

```bash
mergekit-extract-lora --model finetuned_model --base-model base_model --out-path lora_output
```

### Mixture of Experts Merging

```bash
mergekit-moe --config moe_config.yml --out-dir moe_model
```

### Evolutionary Optimization

```bash
mergekit-evolve --config evo_config.yml --out-dir optimized_model
```

## Internal Pipeline Execution

When using `mergekit-yaml` (or via `run_merge.py`), the following steps occur:

1. **Configuration Loading**: The YAML file is parsed into a `MergeConfiguration` object
2. **Architecture Analysis**: Models are inspected to ensure compatibility
3. **Merge Planning**: `MergePlanner` class creates a plan for merging each tensor:
   - Normalizes configuration (converts models→slices if needed)
   - Creates tasks for each weight tensor
   - Builds a computational graph for execution
4. **Task Execution**: Tasks are executed in dependency order:
   - Tensors are loaded from input models
   - Merge algorithms are applied
   - Results are saved to the output model
5. **Finalization**: 
   - Configuration is updated
   - Tokenizer is built/copied
   - Model card is generated
   - Files are saved to the output directory

## Usage Considerations

- Merging models with different architectures is risky but can be attempted with `--allow-crimes`
- Always evaluate merged models against your specific tasks
- Be mindful of licensing when combining models from different sources
- Consider the ethical implications of combining models with different alignment properties

## References

- [MergeKit GitHub Repository](https://github.com/arcee-ai/mergekit)
- [Task Arithmetic Paper](https://arxiv.org/abs/2212.04089)
- [TIES Paper](https://arxiv.org/abs/2306.01708)
- [DARE Paper](https://arxiv.org/abs/2311.03099)
- [Model Breadcrumbs Paper](https://arxiv.org/abs/2312.06795)
- [DELLA Paper](https://arxiv.org/abs/2406.11617) 