import argparse
import yaml
import torch
from safetensors.torch import load_file, safe_open
from huggingface_hub import hf_hub_download
import os
import json
import numpy as np
from tqdm import tqdm

def rms_to_rms_operator_norm(W: torch.Tensor) -> torch.Tensor:
    """Calculates the RMS to RMS operator norm for a 2D weight tensor."""
    if W.ndim != 2:
        return torch.tensor(float('nan'), device=W.device)
    d_out, d_in = W.shape
    if d_out == 0:
        return torch.tensor(0.0, device=W.device)
    spectral_norm = torch.linalg.norm(W, ord=2)
    return torch.sqrt(torch.tensor(d_in / d_out, device=W.device)) * spectral_norm

def l1_to_rms_operator_norm(W: torch.Tensor) -> torch.Tensor:
    """Calculates the l1 to RMS operator norm for a 2D weight tensor."""
    if W.ndim != 2:
        return torch.tensor(float('nan'), device=W.device)
    d_out, _ = W.shape
    if d_out == 0:
        return torch.tensor(0.0, device=W.device)
    l1_to_l2_norm = torch.linalg.norm(W, ord=2, dim=0).max()
    return (1 / torch.sqrt(torch.tensor(d_out, device=W.device))) * l1_to_l2_norm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LazyModelLoader:
    def __init__(self, repo_id, device):
        self.repo_id = repo_id
        self.archives = {}
        self.weight_map = {}
        self._keys = []
        self.device = device

        try:
            index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
            with open(index_path) as f:
                index = json.load(f)
            self.weight_map = index['weight_map']
            self._keys = list(self.weight_map.keys())
            shard_files = set(self.weight_map.values())
            print(f"  {self.repo_id}: Found sharded model with {len(shard_files)} shards.")
            for shard_file in shard_files:
                shard_path = hf_hub_download(repo_id=repo_id, filename=shard_file)
                # Note: safe_open device is 'cpu' to avoid loading everything to GPU at once
                self.archives[shard_file] = safe_open(shard_path, framework="pt", device="cpu")
        except Exception:
            print(f"  {self.repo_id}: No index found. Assuming single-file model.")
            model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            archive = safe_open(model_path, framework="pt", device="cpu")
            self.archives['model.safetensors'] = archive
            self._keys = archive.keys()
            for key in self._keys:
                self.weight_map[key] = 'model.safetensors'

    def get_tensor(self, key):
        if key not in self.weight_map:
            return None
        shard_name = self.weight_map[key]
        # Move tensor to the target device when it's requested
        return self.archives[shard_name].get_tensor(key).to(self.device)

    def keys(self):
        return self._keys

def save_results(results, param_shapes, output_path):
    if not results:
        print("No results to save.")
        return

    all_params = set()
    for model_results in results.values():
        all_params.update(model_results.keys())
    param_names = sorted(list(all_params - {"Overall Mean"}))
    param_names.append("Overall Mean") # Ensure Overall Mean is the last row
    
    model_names = sorted(results.keys())

    if output_path:
        try:
            import pandas as pd
        except ImportError:
            print("Error: pandas and openpyxl are required for Excel output. Please install them with 'pip install pandas openpyxl'")
            return
        
        # Create a multi-level column index
        stats = ['Hybrid Norm', 'Preconditioner Mean']
        columns = pd.MultiIndex.from_product([model_names, stats],
                                             names=['Model', 'Statistic'])
        df = pd.DataFrame(index=param_names, columns=columns)
        
        # Add shape column
        shapes = [str(param_shapes.get(p, '')) for p in param_names if p != "Overall Mean"]
        shapes.append('') # for Overall Mean row
        df.insert(0, "Shape", shapes)

        for model_name, model_results in results.items():
            for param_name, values in model_results.items():
                if isinstance(values, dict):
                    df.at[param_name, (model_name, 'Hybrid Norm')] = values.get('hybrid_norm')
                    df.at[param_name, (model_name, 'Preconditioner Mean')] = values.get('preconditioner_mean')

        df.index.name = "Parameter"
        df.to_excel(output_path)
        print(f"Results saved to {output_path}")
    else:
        # Console output
        print("Excel output is recommended for detailed analysis. Use the --output flag.")
        for param in param_names:
            shape_str = f" (Shape: {param_shapes.get(param, '')})" if param != "Overall Mean" else ""
            print(f"\n--- Parameter: {param}{shape_str} ---")
            for model in model_names:
                print(f"  - Model: {model}")
                if param in results[model]:
                    values = results[model][param]
                    print(f"    Hybrid Norm:         {values.get('hybrid_norm', 'N/A'):.4e}")
                    print(f"    Preconditioner Mean: {values.get('preconditioner_mean', 'N/A'):.4e}")


def analyze_models(config_path, output_path=None):
    device = get_device()
    print(f"Using device: {device}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_model_name = config.get('base_model')
    if not base_model_name:
        print("Error: 'base_model' not found in config.")
        return

    print("Initializing model loaders...")
    base_loader = LazyModelLoader(base_model_name, device)
    
    specialist_loaders = {
        model_info['model']: LazyModelLoader(model_info['model'], device)
        for model_info in config.get('models', [])
    }

    preconditioner_archives = {}
    for model_info in config.get('models', []):
        model_name = model_info['model']
        path = model_info.get('parameters', {}).get('preconditioner_path')
        if path:
            try:
                repo_id = model_name
                filename = path
                if filename.startswith(repo_id):
                    filename = os.path.relpath(filename, repo_id)
                local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                # Preconditioners are also loaded to CPU first
                preconditioner_archives[model_name] = safe_open(local_path, framework="pt", device="cpu")
            except Exception as e:
                print(f"  Could not open preconditioner for {model_name}: {e}")

    results = {name: {} for name in specialist_loaders.keys()}
    param_shapes = {}
    
    param_keys = base_loader.keys()
    print(f"Analyzing {len(param_keys)} parameters...")

    for key in tqdm(param_keys, desc="Analyzing Tensors"):
        base_tensor = base_loader.get_tensor(key)
        if base_tensor is None:
            continue
        param_shapes[key] = tuple(base_tensor.shape)

        for name, loader in specialist_loaders.items():
            specialist_tensor = loader.get_tensor(key)
            if specialist_tensor is None:
                continue
            
            diff = specialist_tensor - base_tensor.to(specialist_tensor.device, dtype=specialist_tensor.dtype)
            diff_float32 = diff.to(torch.float32)
            hybrid_norm = float('nan')

            if key in ["lm_head.weight", "model.embed_tokens.weight"]:
                hybrid_norm = l1_to_rms_operator_norm(diff_float32).item()
            elif diff.ndim >= 2:
                hybrid_norm = rms_to_rms_operator_norm(diff_float32).item()
            elif diff.ndim == 1:
                hybrid_norm = torch.sqrt(torch.mean(diff.pow(2))).item()
            
            preconditioner_mean = float('nan')
            if name in preconditioner_archives:
                pc_archive = preconditioner_archives[name]
                if key in pc_archive.keys():
                    preconditioner_mean = pc_archive.get_tensor(key).to(device).mean().item()

            results[name][key] = {
                'hybrid_norm': hybrid_norm,
                'preconditioner_mean': preconditioner_mean
            }

    print("Calculating overall means...")
    for name, model_results in results.items():
        if model_results:
            # Filter out NaNs from hybrid norms before calculating mean
            hybrid_norms = [
                v['hybrid_norm'] for v in model_results.values() 
                if isinstance(v, dict) and 'hybrid_norm' in v and not np.isnan(v['hybrid_norm'])
            ]
            if hybrid_norms:
                 model_results["Overall Mean"] = { 'hybrid_norm': np.mean(hybrid_norms) }

    save_results(results, param_shapes, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze mergekit models.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--output", "-o", type=str, help="Path to save the output Excel file.")
    args = parser.parse_args()
    
    analyze_models(args.config_file, args.output) 