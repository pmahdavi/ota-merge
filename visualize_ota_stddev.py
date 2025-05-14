import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np # For log10 in histogram
import argparse
import os
import sys
import glob

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Epsilon for preventing division by zero in ratio calculation
DIV_EPSILON = 1e-10
# Epsilon for LogNorm to prevent issues with zero/negative values if they somehow occur
LOGNORM_EPSILON = 1e-10 
# Threshold for Max/Mean Ratio visualization
MAX_MEAN_THRESHOLD = 6 # User updated value

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

    plt.title(f"Histogram of {title_prefix}\\n{os.path.basename(base_filename)}")
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
    plt.title(f"Heatmap of {title_prefix}\\n{os.path.basename(base_filename)}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 0")
    clean_title_prefix = title_prefix.lower().replace(' ', '_').replace('/','_').replace('(','').replace(')','').replace('>','gt')
    heatmap_path = os.path.join(log_dir_path, f"{base_filename}_{clean_title_prefix}_heatmap_{scale_type}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  Saved heatmap to: {heatmap_path}")

def plot_dominant_model_heatmap(display_tensor: torch.Tensor, num_models: int, title_prefix: str, base_filename: str, log_dir_path: str, threshold_value: float):
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
    tick_labels = [f'< {threshold_value:.1f}'] + [f'Model {i}' for i in range(num_models)]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label(f"Dominant Model Index (Max/Mean Ratio \u2265 {threshold_value:.1f})")

    plt.title(f"{title_prefix}\\n{os.path.basename(base_filename)}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 0")
    clean_title_prefix = title_prefix.lower().replace(' ', '_').replace('/','_').replace('(','').replace(')','').replace('>','gt')
    heatmap_path = os.path.join(log_dir_path, f"{base_filename}_{clean_title_prefix}_dominant_model_heatmap_thresh{threshold_value}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"  Saved dominant model heatmap to: {heatmap_path}")

def analyze_and_visualize_tensor_metrics(log_dir_path: str, tensor_base_name: str):
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

    # --- Process Max/Min & Max/Mean Weight Ratios --- 
    weights_stack_file_name = f"{tensor_base_name}_weights_stack.pt"
    weights_stack_file_path = os.path.join(log_dir_path, weights_stack_file_name)

    if os.path.exists(weights_stack_file_path):
        print(f"Loading weights_stack tensor from: {weights_stack_file_path}")
        try:
            weights_stack_tensor = torch.load(weights_stack_file_path, map_location=device).float()
            num_models = weights_stack_tensor.shape[0]
            if num_models <= 1:
                print(f"  Weights stack for {tensor_base_name} has only one model ({num_models}), skipping ratio analyses.")
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
                plot_dominant_model_heatmap(dominant_model_display_tensor, num_models, title_max_mean_dom_model_map, tensor_base_name, log_dir_path, MAX_MEAN_THRESHOLD)

        except Exception as e:
            print(f"Error processing weights_stack tensor {weights_stack_file_path}: {e}", file=sys.stderr)
    else:
        print(f"Weights_stack tensor file not found: {weights_stack_file_path}, skipping ratio analyses.")
    print("---")

def process_all_tensors_in_dir(log_dir_path: str):
    """
    Scans a directory for all *_elementwise_std.pt files and visualizes metrics for each corresponding tensor.
    """
    if not os.path.isdir(log_dir_path):
        print(f"Error: Log directory not found: {log_dir_path}", file=sys.stderr)
        sys.exit(1)

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
            analyze_and_visualize_tensor_metrics(log_dir_path, tensor_base_name)
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