# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install in development mode with all dependencies
pip install -e ".[dev,test]"

# Install basic package only
pip install -e .
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_ota.py

# Run tests with specific pattern
python -m pytest tests/ -k "ota"
```

### Code Quality
```bash
# Run pre-commit hooks (formatting, linting, validation)
pre-commit run --all-files

# Auto-format code with Black
black mergekit/

# Sort imports
isort mergekit/
```

### Main CLI Tools
```bash
# Primary merge tool using YAML configuration
mergekit-yaml config.yml output-dir [--cuda] [--lazy-unpickle]

# Other merge interfaces
mergekit-legacy       # Legacy command-line interface
mergekit-pytorch      # Direct PyTorch model merging
mergekit-multi        # Multi-model merging

# Specialized tools
mergekit-extract-lora      # Extract LoRA from fine-tuned model
mergekit-moe              # Create Mixture of Experts models
mergekit-layershuffle     # Shuffle/rearrange model layers
mergekit-tokensurgeon     # Tokenizer manipulation operations
mergekit-evolve           # Evolutionary merge optimization
```

### PBS Job Submission (run_merge.py)
```bash
# Submit a merge job to PBS cluster
python run_merge.py config.yml [options]

# Options:
--output_dir PATH      # Base output directory (default: /scratch/pxm5426/runs/lora-exploration/merged_models)
--walltime TIME        # PBS walltime (default: 12:00:00)
--ngpus N             # Number of GPUs (default: 2)
--ncpus N             # Number of CPUs (default: 16)
--mem SIZE            # Memory allocation (default: 80g)
--job_name NAME       # PBS job name (default: mergekit)
--dry-run             # Show PBS script without submitting
--no_allow_crimes     # Disable experimental features
--no_trust_remote_code # Disable remote code execution
```

The `run_merge.py` script automatically:
- Parses YAML configs to generate descriptive output directory names
- Creates PBS scripts with proper resource allocation
- Handles method-specific naming (e.g., `ota_math-coding_pow0.5_precond-thresh1e-14`)
- Sets up conda environment and CUDA access
- Tracks jobs with email notifications

## Architecture Overview

MergeKit is a sophisticated toolkit for merging pre-trained language models using various mathematical approaches. The codebase follows a modular, task-based architecture designed for memory efficiency and extensibility.

### Core Design Principles

1. **Task-Based Execution**: Uses a computational graph with `Task` objects that define tensor operations. Tasks are executed lazily with smart caching and resource management.

2. **Lazy Loading**: Implements lazy tensor loading through the `LazyTensorLoader` system to handle large models with limited memory. Tensors are only loaded when needed.

3. **Plugin Architecture**: Merge methods can be added as plugins using two approaches:
   - **Static Registration**: Register in `mergekit/merge_methods/registry.py` for complex methods
   - **Decorator API**: Use `@MergeMethod.register("method_name")` in `easy_define.py` for simple methods

4. **Configuration-Driven**: Uses Pydantic models for type-safe YAML configuration with hierarchical parameter specification and validation.

### Key Components

- **`mergekit/merge_methods/`**: Merge algorithm implementations
  - `registry.py`: Static method registration
  - `easy_define.py`: Decorator-based method registration
  - Each method inherits smart memory management and device handling
  - Supports GPU acceleration with automatic CPU fallback
  
- **`mergekit/_data/architectures/`**: Model architecture definitions (JSON)
  - Supports 40+ architectures including Llama, Mistral, GPT variants, Gemma, Phi, Qwen, etc.
  - Handles architecture-specific tensor naming and transformations

- **`mergekit/io/`**: Tensor I/O system
  - `LazyTensorLoader`: Deferred loading of tensors
  - `TensorWriter`: Efficient tensor serialization with progress tracking

- **`mergekit/graph.py`**: Task execution engine
  - Manages dependencies between tensor operations
  - Implements caching and resource cleanup

### Available Merge Methods

**Basic Interpolation Methods:**
- `linear` - Linear interpolation between models
- `slerp` - Spherical linear interpolation
- `nuslerp` - Normalized unit SLERP
- `multislerp` - Multiple SLERP operations
- `passthrough` - Pass through without modification

**Task Vector Methods:**
- `task_arithmetic` - Operates on task vectors (model - base_model)
- `ties` - TIES (Task-wise Importance Estimation)
- `dare_ties` - DARE with TIES sparsification
- `dare_linear` - Linear DARE method

**Advanced Geometric Methods:**
- `model_stock` - Model Stock averaging
- `arcee_fusion` - Arcee Fusion merge
- `karcher` - Karcher mean on manifold
- `ota` - Optimization Trajectory Aware (uses optimizer states)
- `della` - DELLA method
- `della_linear` - Linear variant of DELLA
- `sce` - Sign Consensus Elimination

**Specialized Methods:**
- `breadcrumbs` - Model Breadcrumbs tracking
- `breadcrumbs_ties` - Breadcrumbs with TIES
- `nearswap` - Near swap method

### OTA (Optimization Trajectory Aware) Merging

OTA is a novel merging method that leverages Adam optimizer's second moments (exp_avg_sq) as a curvature approximation for improved merge quality. This method is based on the insight that optimizer states contain valuable information about the loss landscape curvature.

**Key Concepts:**
- Uses Adam's second moment tensors as preconditioners
- Performs element-wise weighted averaging with curvature awareness
- Supports various normalization modes for robustness

**Configuration Parameters:**
- `preconditioner_path`: Path to exported Adam second moments (required per model)
- `power`: Power to raise the preconditioner values (default: 0.5)
- `epsilon`: Small value for numerical stability (default: 1e-16)
- `precond_threshold`: Mask elements where preconditioner < threshold (optional)
- `normalise`: Normalization mode - `none`, `global`, `layer`, `inverse` (default: none)

**Example OTA Configuration:**
```yaml
merge_method: ota

parameters:
  epsilon: 1e-16
  normalise: none
  power: 0.5
  precond_threshold: 1e-14

models:
  - model: pmahdavi/Llama-3.1-8B-math-reasoning
    parameters:
      preconditioner_path: pmahdavi/Llama-3.1-8B-math-reasoning/export/exp_avg_sq.safetensors
  - model: pmahdavi/Llama-3.1-8B-coding
    parameters:
      preconditioner_path: pmahdavi/Llama-3.1-8B-coding/export/exp_avg_sq.safetensors

dtype: bfloat16
tokenizer_source: union
```

**Implementation Details:**
- Implemented in `mergekit/merge_methods/ota.py`
- Supports fallback to linear merging when preconditioners unavailable
- Handles sparse preconditioners efficiently
- Comprehensive tests in `tests/test_ota.py`

### Configuration System

YAML configurations support:
- Model selection with per-layer slicing ("Frankenmerging")
- Hierarchical parameter specification with tensor name filters
- Method-specific parameters (e.g., `preconditioner_path` for OTA)
- Gradient parameters for smooth interpolation
- Tokenizer handling options (`union`, specific source selection)
- Data type specification (`float16`, `bfloat16`, `float32`)

Example configuration structure:
```yaml
models:
  - model: path/to/model1
    parameters:
      weight: 0.5
slices:
  - sources:
      - model: 0
        layer_range: [0, 16]
merge_method: ota
parameters:
  preconditioner_path: path/to/preconditioner
base_model: path/to/base
tokenizer_source: union
dtype: bfloat16
```

### Recent Research Focus

Active development on OTA-Merging (NeurIPS paper) which leverages Adam optimizer's second moments to approximate curvature for better merge quality. The method shows promising results in preserving specialized capabilities while merging models trained on different tasks (e.g., math reasoning + coding + instruction following).