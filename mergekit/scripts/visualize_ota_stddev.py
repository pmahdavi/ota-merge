import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np # For log10 in histogram
import argparse
import os
import sys
import glob
import json # Added for model manifest
import re # For specific name extraction

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Epsilon for preventing division by zero in ratio calculation
DIV_EPSILON = 1e-10
# Epsilon for LogNorm to prevent issues with zero/negative values if they somehow occur
LOGNORM_EPSILON = 1e-10 
# Threshold for Max/Mean Ratio visualization
MAX_MEAN_THRESHOLD = 1.25 # Set to 1.7 for 70% dominance relative to mean

def _extract_capability_name(full_name: str) -> str:
    """Extracts a preferred short name, prioritizing 'capability' from 'mixture_capability'."""
    base_name = os.path.basename(full_name) 
    
    # Patterns to find 'mixture_capability' or 'Llama-3.1-8B_tulu3_capability'
    # The key is to capture the part *after* 'mixture_'
    patterns = [
        r"mixture_([a-zA-Z0-9_]+?)_full", # e.g., mixture_coding_full -> coding
        r"mixture_([a-zA-Z0-9_]+)",      # e.g., mixture_math -> math
        # Fallback patterns if the above don't match, looking for capability after Llama-3.1-8B_tulu3_
        # These are less ideal if you specifically want what was after "mixture_"
        # but can be a fallback if "mixture_" is missing but the Llama prefix is there.
        r"Llama-3\.1-8B_tulu3_([a-zA-Z0-9_]+?)_full",
        r"Llama-3\.1-8B_tulu3_([a-zA-Z0-9_]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base_name)
        if match:
            # The first captured group (match.group(1)) will be the desired capability part
            # e.g., for "mixture_coding_full", group(1) is "coding"
            # for "mixture_math", group(1) is "math"
            extracted_name = match.group(1)
            return extracted_name

    # Fallback if no specific pattern matches, use basename and then truncate if still too long/path-like
    # This part remains as a general fallback.
    if len(base_name) > 25 and ("/" in base_name or "\\" in base_name):
        return "... " + base_name[-22:]
    elif len(base_name) > 25 :
        return base_name[:22] + "..."
    return base_name

def plot_histogram(data_tensor: torch.Tensor, title_prefix: str, base_filename: str, log_dir_path: str, use_log_x_scale_heuristic: bool = False, force_linear_x_scale: bool = False):
    """Helper function to generate and save a histogram."""
    plt.figure(figsize=(10, 6))
    numpy_data = data_tensor.cpu().flatten().numpy()
    current_xlabel = title_prefix
    positive_data_for_log = numpy_data[numpy_data > 0]

    if force_linear_x_scale:
        plt.hist(numpy_data, bins=100)
        print(f"  Forcing linear scale for {title_prefix} histogram x-axis.")
    elif use_log_x_scale_heuristic and len(positive_data_for_log) > 0 and positive_data_for_log.max() > 1000:
        if (numpy_data == 0).any(): 
            min_log_val = np.log10(max(LOGNORM_EPSILON, positive_data_for_log.min())) 
            max_log_val = np.log10(positive_data_for_log.max())
            if max_log_val > min_log_val:
                bins = np.logspace(min_log_val, max_log_val, 50)
                plt.hist(positive_data_for_log, bins=bins, label=f'>0 values (max {positive_data_for_log.max():.2e})')
            else: 
                plt.hist(positive_data_for_log, bins=50, label=f'>0 values (max {positive_data_for_log.max():.2e})')
            plt.legend()
            plt.xscale('log')
        else:
            min_log_val = np.log10(max(LOGNORM_EPSILON, positive_data_for_log.min()))
            max_log_val = np.log10(positive_data_for_log.max())
            if max_log_val > min_log_val:
                 bins = np.logspace(min_log_val, max_log_val, 50)
                 plt.hist(positive_data_for_log, bins=bins)
            else:
                 plt.hist(positive_data_for_log, bins=50)
            plt.xscale('log')
        current_xlabel = f"{title_prefix} (Log Scale for x > 0)"
        print(f"  Using log scale for {title_prefix} histogram x-axis (for values > 0).")
    else:
        plt.hist(numpy_data, bins=100)
        print(f"  Using linear scale for {title_prefix} histogram x-axis.")

    plt.title(f"Histogram of {title_prefix}\n{os.path.basename(base_filename)}")
    plt.xlabel(current_xlabel)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    clean_title_prefix = title_prefix.lower().replace(' ', '_').replace('/','_').replace('(','').replace(')','').replace('>','gt')
    histogram_path = os.path.join(log_dir_path, f"{base_filename}_{clean_title_prefix}_histogram.png")
    plt.savefig(histogram_path)
    plt.close()
    print(f"  Saved histogram to: {histogram_path}")

def plot_heatmap(data_tensor: torch.Tensor, title_prefix: str, base_filename: str, log_dir_path: str, force_linear_scale: bool = False):
    """Helper function to generate and save a heatmap for continuous data if data is 2D."""
    if data_tensor.ndim != 2:
        print(f"  Tensor for {title_prefix} is not 2D (shape: {data_tensor.shape}), skipping heatmap.")
        return

    plt.figure(figsize=(12, 10))
    numpy_tensor = data_tensor.cpu().numpy()
    norm = None
    scale_type = "linear"
    imshow_vmin = None
    imshow_vmax = None

    if force_linear_scale:
        print(f"  Forcing linear scale for {title_prefix} heatmap.")
        imshow_vmin = 0 
        imshow_vmax = numpy_tensor.max() if numpy_tensor.size > 0 else 1.0 
        if imshow_vmax <= imshow_vmin : 
            imshow_vmax = imshow_vmin + 1.0 
    else:
        positive_values = numpy_tensor[numpy_tensor > LOGNORM_EPSILON]
        min_positive_val_for_norm = positive_values.min() if len(positive_values) > 0 else LOGNORM_EPSILON
        max_val_for_norm = positive_values.max() if len(positive_values) > 0 else LOGNORM_EPSILON * 10

        if len(positive_values) > 0 and max_val_for_norm > min_positive_val_for_norm * 100: 
            norm = mcolors.LogNorm(vmin=max(min_positive_val_for_norm, LOGNORM_EPSILON), vmax=max_val_for_norm)
            scale_type = "logscale"
            print(f"  Using LogNorm for {title_prefix} heatmap (for values > {LOGNORM_EPSILON:.1e}): vmin={max(min_positive_val_for_norm, LOGNORM_EPSILON):.2e}, vmax={max_val_for_norm:.2e}")
        else:
            print(f"  Using linear scale for {title_prefix} heatmap.")
            imshow_vmin = numpy_tensor.min() if numpy_tensor.size > 0 else 0.0
            imshow_vmax = numpy_tensor.max() if numpy_tensor.size > 0 else 1.0
            if imshow_vmax <= imshow_vmin:
                imshow_vmax = imshow_vmin + 1.0

    aspect_ratio = numpy_tensor.shape[1] / numpy_tensor.shape[0]
    aspect = 'auto' if aspect_ratio > 10 or aspect_ratio < 0.1 else 'equal'
    display_tensor = numpy_tensor 
    im = plt.imshow(display_tensor, aspect=aspect, cmap='viridis', norm=norm, vmin=imshow_vmin, vmax=imshow_vmax)
    plt.colorbar(im, label=title_prefix)
    plt.title(f"Heatmap of {title_prefix}\n{os.path.basename(base_filename)}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 0")
    clean_title_prefix = title_prefix.lower().replace(' ', '_').replace('/','_').replace('(','').replace(')','').replace('>','gt')
    heatmap_path = os.path.join(log_dir_path, f"{base_filename}_{clean_title_prefix}_heatmap_{scale_type}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  Saved heatmap to: {heatmap_path}")

def plot_single_model_weight_heatmap(
    tensor: torch.Tensor, 
    model_idx: int, 
    base_filename: str, 
    log_dir_path: str, 
    model_names: list[str] | None = None, # New parameter
    max_side: int = 256 # Max side for downsampling, can be adjusted
):
    """Plot a heatmap for a single model's 2D weight tensor (absolute values, log10 scale)."""
    if tensor.numel() == 0:
        print(f"  Tensor for model {model_idx} ({base_filename}) is empty, skipping heatmap.")
        return
    if tensor.ndim != 2:
        print(f"  Tensor for model {model_idx} ({base_filename}) is not 2D (shape: {tensor.shape}), skipping heatmap.")
        return
        
    data = tensor.detach().abs().cpu() # Use absolute values for log scale

    # Down‑sample only if *both* dimensions are substantially larger than max_side
    # This heuristic might need adjustment based on typical tensor sizes.
    # Using a simpler downsampling if any dimension is too large.
    if data.shape[0] > max_side or data.shape[1] > max_side:
        data = _downsample_tensor_2d_viz(data, max_side=max_side)
        print(f"  Downsampled model {model_idx} ({base_filename}) weights to {data.shape} for heatmap.")

    arr = data.numpy()
    # Add small epsilon to prevent log(0) or log(negative if abs() wasn't used)
    # LOGNORM_EPSILON is already defined globally
    eps = LOGNORM_EPSILON 
    
    plt.figure(figsize=(6, 5)) # Adjusted figure size slightly
    # Use np.maximum with eps to handle exact zeros if abs() wasn't applied or if original data could be zero
    img = plt.imshow(np.log10(np.maximum(arr, eps)), cmap='viridis', aspect='auto') 
    
    raw_model_name = model_names[model_idx] if model_names and model_idx < len(model_names) else f'Model {model_idx}'
    model_display_name = _extract_capability_name(raw_model_name)
   
    plot_title = f"{model_display_name} Weights\n{os.path.basename(base_filename)}"
    plt.title(plot_title)
    plt.axis("off")
    cbar = plt.colorbar(img, fraction=0.046, pad=0.04)
    cbar.set_label("log10(|weight value|)") # Label reflects absolute value
    plt.tight_layout()
    
    # Consistent file naming
    # base_filename is already sanitized (e.g. model-layers-0-...)
    heatmap_filename = f"{base_filename}_model_{model_idx}_{model_display_name.replace(' ', '-')}_weights_heatmap.png"
    save_path = os.path.join(log_dir_path, heatmap_filename)
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {model_display_name} weight heatmap to: {save_path}")

def plot_dominant_model_heatmap(display_tensor: torch.Tensor, num_models: int, title_prefix: str, base_filename: str, log_dir_path: str, threshold_value: float, model_names: list[str] | None = None):
    """Plots a categorical heatmap showing the dominant model index."""
    if display_tensor.ndim != 2:
        print(f"  Tensor for {title_prefix} (dominant model) is not 2D (shape: {display_tensor.shape}), skipping heatmap.")
        return

    plt.figure(figsize=(12, 10))
    numpy_display_tensor = display_tensor.cpu().numpy()

    # Define colors: black for below threshold (-1), then distinct colors for models 0 to N-1
    # Using a more vibrant and distinct colormap like Set1 or qualitative ones
    if num_models <= 10:
        model_colors = plt.cm.get_cmap('tab10', num_models).colors
    elif num_models <= 12:
        model_colors = plt.cm.get_cmap('Paired', num_models).colors 
    elif num_models <= 20:
        model_colors = plt.cm.get_cmap('tab20', num_models).colors
    else: # Fallback for more models, though distinctness suffers
        model_colors = plt.cm.get_cmap('viridis', num_models).colors 
        print(f"Warning: Number of models ({num_models}) is large for distinct categorical colors. Using viridis.")

    colors = ['black'] + [mcolors.to_hex(c) for c in model_colors]
    cmap = mcolors.ListedColormap(colors)
    
    # Bounds for BoundaryNorm: -1.5, -0.5, 0.5, ..., num_models-0.5
    bounds = [-1.5] + [i - 0.5 for i in range(num_models + 1)]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    aspect_ratio = numpy_display_tensor.shape[1] / numpy_display_tensor.shape[0]
    aspect = 'auto' if aspect_ratio > 10 or aspect_ratio < 0.1 else 'equal'

    im = plt.imshow(numpy_display_tensor, aspect=aspect, cmap=cmap, norm=norm)
    
    # Configure colorbar
    # Ticks should be in the middle of each color band: -1, 0, 1, ..., num_models-1
    ticks = list(range(-1, num_models))
    cbar = plt.colorbar(im, ticks=ticks, spacing='proportional')
    
    base_tick_labels = [f'< {threshold_value:.1f}']
    for i in range(num_models):
        raw_model_name = model_names[i] if model_names and i < len(model_names) else f'Model {i}'
        model_display_name = _extract_capability_name(raw_model_name)
        # Shorter name for colorbar directly, _extract_capability_name handles length now
        if len(model_display_name) > 15: # Specific shorter limit for colorbar labels
             model_display_name = model_display_name[:12] + "..." 
        base_tick_labels.append(model_display_name)

    cbar.set_ticklabels(base_tick_labels)
    cbar.set_label(f"Dominant Model Index (Max/Mean Ratio \u2265 {threshold_value:.1f})")

    plt.title(f"{title_prefix}\n{os.path.basename(base_filename)}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 0")
    clean_title_prefix = title_prefix.lower().replace(' ', '_').replace('/','_').replace('(','').replace(')','').replace('>','gt')
    heatmap_path = os.path.join(log_dir_path, f"{base_filename}_{clean_title_prefix}_dominant_model_heatmap_thresh{threshold_value}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  Saved dominant model heatmap to: {heatmap_path}")

# --- NEW --- Function adapted from visualize_optim_state.py
def _downsample_tensor_2d_viz(t: torch.Tensor, max_side: int = 256) -> torch.Tensor: # Renamed to avoid conflict if other utils exist
    """Down‑sample a 2‑D tensor so that both sides are ≤ ``max_side``."""
    h, w = t.shape
    if h <= max_side and w <= max_side:
        return t
    step_h = max(1, int(torch.ceil(torch.tensor(h / max_side)).item()))
    step_w = max(1, int(torch.ceil(torch.tensor(w / max_side)).item()))
    return t[::step_h, ::step_w]

def analyze_and_visualize_tensor_metrics(log_dir_path: str, tensor_base_name: str, model_names: list[str] | None = None):
    """
    Loads stddev and weights_stack tensors, then calculates and visualizes
    standard deviation, max/min weight ratio, and thresholded max/mean weight ratio (with dominant model heatmap).
    All tensor computations are attempted on GPU if available.
    """
    print(f"Processing tensor base: {tensor_base_name}")

    # --- Process Standard Deviation --- 
    stddev_file_name = f"{tensor_base_name}_elementwise_std.pt"
    stddev_file_path = os.path.join(log_dir_path, stddev_file_name)
    
    if os.path.exists(stddev_file_path):
        print(f"Loading stddev tensor from: {stddev_file_path}")
        try:
            stddev_tensor = torch.load(stddev_file_path, map_location=device).float()
            print(f"  StdDev Tensor stats: Min={stddev_tensor.min().item():.4e}, Max={stddev_tensor.max().item():.4e}, Mean={stddev_tensor.mean().item():.4e}, Std={stddev_tensor.std().item():.4e}")
            plot_histogram(stddev_tensor, "Element-wise StdDev", tensor_base_name, log_dir_path)
            plot_heatmap(stddev_tensor, "Element-wise StdDev", tensor_base_name, log_dir_path)
        except Exception as e:
            print(f"Error processing stddev tensor {stddev_file_path}: {e}", file=sys.stderr)
    else:
        print(f"Stddev tensor file not found: {stddev_file_path}, skipping stddev analysis.")

    # --- Process Max/Min & Max/Mean Weight Ratios & Individual Model Weights --- 
    weights_stack_file_name = f"{tensor_base_name}_weights_stack.pt"
    weights_stack_file_path = os.path.join(log_dir_path, weights_stack_file_name)

    if os.path.exists(weights_stack_file_path):
        print(f"Loading weights_stack tensor from: {weights_stack_file_path}")
        try:
            weights_stack_tensor = torch.load(weights_stack_file_path, map_location=device).float()
            num_models = weights_stack_tensor.shape[0]

            # --- NEW: Plot individual model weight heatmaps ---
            if weights_stack_tensor.ndim == 3: # Expects [num_models, H, W]
                print(f"  Generating heatmaps for individual model weights (from {weights_stack_file_name})...")
                for i in range(num_models):
                    model_weight_tensor = weights_stack_tensor[i, :, :]
                    plot_single_model_weight_heatmap(
                        tensor=model_weight_tensor,
                        model_idx=i,
                        base_filename=tensor_base_name, # Pass the original base name
                        log_dir_path=log_dir_path,
                        model_names=model_names # Pass loaded model names
                    )
            elif weights_stack_tensor.ndim > 1 : # If it's not 3D but still multi-dim (e.g. for biases [num_models, Features])
                 print(f"  Weights stack for {tensor_base_name} is {weights_stack_tensor.ndim}D. Individual model heatmaps are only generated for 3D stacks (representing 2D weights per model).")
            # --- END NEW ---


            if num_models <= 1:
                print(f"  Weights stack for {tensor_base_name} has {num_models} model(s), skipping ratio analyses.")
            else:
                max_weights_op = torch.max(weights_stack_tensor, dim=0)
                max_weights = max_weights_op.values
                max_indices = max_weights_op.indices # Get indices of max elements
                min_weights = torch.min(weights_stack_tensor, dim=0).values
                mean_weights = torch.mean(weights_stack_tensor, dim=0)
                
                # Max/Min Ratio (still uses heuristic for log scale for heatmap)
                max_min_ratio_tensor = max_weights / (min_weights + DIV_EPSILON)
                max_min_ratio_tensor = torch.clamp(max_min_ratio_tensor, max=1e12)
                max_min_ratio_tensor = torch.nan_to_num(max_min_ratio_tensor, nan=0.0)
                print(f"  Max/Min Ratio Tensor stats: Min={max_min_ratio_tensor.min().item():.4e}, Max={max_min_ratio_tensor.max().item():.4e}, Mean={max_min_ratio_tensor.mean().item():.4e}, Std={max_min_ratio_tensor.std().item():.4e}")
                plot_histogram(max_min_ratio_tensor, "Max-Min Weight Ratio", tensor_base_name, log_dir_path, use_log_x_scale_heuristic=True)
                plot_heatmap(max_min_ratio_tensor, "Max-Min Weight Ratio", tensor_base_name, log_dir_path) 

                # Max/Mean Ratio (Thresholded values for histogram, Dominant Model for heatmap)
                title_max_mean_hist = f"Max-Mean Weight Ratio (Values Thresholded > {MAX_MEAN_THRESHOLD})"
                title_max_mean_dom_model_map = f"Dominant Model (Max-Mean Ratio > {MAX_MEAN_THRESHOLD})"
                
                max_mean_ratio_values = max_weights / (mean_weights + DIV_EPSILON)
                
                orig_max_mean_min, orig_max_mean_max, orig_max_mean_mean, orig_max_mean_std = max_mean_ratio_values.min().item(), max_mean_ratio_values.max().item(), max_mean_ratio_values.mean().item(), max_mean_ratio_values.std().item()
                print(f"  Original Max/Mean Ratio stats: Min={orig_max_mean_min:.4e}, Max={orig_max_mean_max:.4e}, Mean={orig_max_mean_mean:.4e}, Std={orig_max_mean_std:.4e}")
                
                # For histogram: threshold values
                max_mean_ratio_values_for_hist = max_mean_ratio_values.clone() # Clone for histogram modification
                max_mean_ratio_values_for_hist[max_mean_ratio_values_for_hist < MAX_MEAN_THRESHOLD] = 0.0
                max_mean_ratio_values_for_hist = torch.clamp(max_mean_ratio_values_for_hist, max=1e12) 
                max_mean_ratio_values_for_hist = torch.nan_to_num(max_mean_ratio_values_for_hist, nan=0.0)
                print(f"  {title_max_mean_hist} Tensor stats: Min={max_mean_ratio_values_for_hist.min().item():.4e}, Max={max_mean_ratio_values_for_hist.max().item():.4e}, Mean={max_mean_ratio_values_for_hist.mean().item():.4e}, Std={max_mean_ratio_values_for_hist.std().item():.4e}")
                plot_histogram(max_mean_ratio_values_for_hist, title_max_mean_hist, tensor_base_name, log_dir_path, force_linear_x_scale=True)

                # For dominant model heatmap: create display tensor with indices
                dominant_model_display_tensor = torch.full_like(max_indices, -1, dtype=torch.long) # -1 for below threshold
                above_threshold_mask = max_mean_ratio_values >= MAX_MEAN_THRESHOLD
                dominant_model_display_tensor[above_threshold_mask] = max_indices[above_threshold_mask]
                plot_dominant_model_heatmap(dominant_model_display_tensor, num_models, title_max_mean_dom_model_map, tensor_base_name, log_dir_path, MAX_MEAN_THRESHOLD, model_names=model_names) # Pass model_names

        except Exception as e:
            print(f"Error processing weights_stack tensor {weights_stack_file_path}: {e}", file=sys.stderr)
    else:
        print(f"Weights_stack tensor file not found: {weights_stack_file_path}, skipping ratio analyses and individual model heatmaps.")
    print("---")

def process_all_tensors_in_dir(log_dir_path: str):
    """
    Scans a directory for all *_elementwise_std.pt files and visualizes metrics for each corresponding tensor.
    """
    if not os.path.isdir(log_dir_path):
        print(f"Error: Log directory not found: {log_dir_path}", file=sys.stderr)
        sys.exit(1)

    # --- Load Model Manifest --- 
    model_names = None
    manifest_path = os.path.join(log_dir_path, "model_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f_manifest:
                model_names = json.load(f_manifest)
            if isinstance(model_names, list) and all(isinstance(name, str) for name in model_names):
                print(f"Successfully loaded model manifest: {model_names}")
            else:
                print(f"Warning: model_manifest.json does not contain a list of strings. Content: {model_names}")
                model_names = None # Invalidate if format is incorrect
        except Exception as e:
            print(f"Warning: Could not load or parse model_manifest.json: {e}")
            model_names = None # Fallback if error
    else:
        print("Model manifest (model_manifest.json) not found. Heatmap titles and labels will use default 'Model X' names.")
    # --- End Load Model Manifest ---

    # Use _elementwise_std.pt to discover base tensor names
    search_pattern = os.path.join(log_dir_path, "*_elementwise_std.pt")
    stddev_files = glob.glob(search_pattern)

    if not stddev_files:
        print(f"No '*_elementwise_std.pt' files found in {log_dir_path} to identify tensors.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(stddev_files)} base tensors to process in {log_dir_path}")
    print(f"Plots will be saved to: {log_dir_path}\n")

    for stddev_file_path in stddev_files:
        file_name_only = os.path.basename(stddev_file_path)
        suffix_to_remove = "_elementwise_std.pt"
        if file_name_only.endswith(suffix_to_remove):
            tensor_base_name = file_name_only[:-len(suffix_to_remove)]
            analyze_and_visualize_tensor_metrics(log_dir_path, tensor_base_name, model_names) # Pass model_names
        else:
            # This case should ideally not be reached if glob pattern is specific
            print(f"Warning: File {file_name_only} does not match expected suffix '{suffix_to_remove}', skipping.", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and visualize OTA tensor metrics (stddev and max/min weight ratio)."
    )
    parser.add_argument(
        "log_dir_path",
        type=str,
        help="Path to the directory containing _elementwise_std.pt and _weights_stack.pt files. Plots will also be saved here.",
    )
    args = parser.parse_args()

    process_all_tensors_in_dir(args.log_dir_path)
    print("Tensor metrics analysis and visualization script finished for all tensors.") 