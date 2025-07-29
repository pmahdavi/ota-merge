#!/usr/bin/env python3
"""
MergeKit Job Submission Script
This script parses YAML config files and launches PBS jobs for merging LLM models.
"""

import os
import sys
import yaml
import socket
import argparse
import subprocess
import time
import uuid
import platform
from pathlib import Path

def create_pbs_script(config_file, merge_method, output_dir, walltime, ngpus, ncpus, mem, job_name, allow_crimes=True, trust_remote_code=True):
    """Create a PBS script for the mergekit job."""
    
    # Get the current hostname
    hostname = socket.gethostname()
    
    # Append domain if it's not already included
    if "." not in hostname:
        hostname = f"{hostname}.eecscl.psu.edu"
    
    # Set command line options based on flags
    options = "--cuda"
    if allow_crimes:
        options += " --allow-crimes"
    if trust_remote_code:
        options += " --trust-remote-code"
    
    pbs_script = f"""#!/bin/tcsh
#PBS -l ngpus={ngpus}
#PBS -l ncpus={ncpus}
#PBS -l walltime={walltime}
#PBS -q workq@{hostname}
#PBS -N {job_name}
#PBS -M pxm5426@psu.edu
#PBS -m bea
#PBS -l mem={mem}
#PBS -o pbs_results/
#PBS -e pbs_results/
cd $PBS_O_WORKDIR

# Source the user's tcsh configuration file
if (-e ~/.tcshrc) then
    source ~/.tcshrc
else
    echo "Warning: ~/.tcshrc not found in the home directory!"
endif

# Activate the mergekit conda environment
# Note: It's better to source the conda init script and then activate
# to ensure it works reliably in non-interactive shells.
if (-e $HOME/miniconda3/bin/conda) then
    eval `$HOME/miniconda3/bin/conda shell.tcsh hook`
    conda activate mergekit
else
    echo "Warning: Conda not found, attempting to run mergekit directly."
endif


# Set the project for wandb
setenv WANDB_PROJECT "mergekit-llama"

echo "Starting model merge: {merge_method}"
echo "Using config file: {config_file}"
echo "Output directory: {output_dir}"

# Ensure output directory exists
mkdir -p {output_dir}

# Run mergekit with the config file
# Usage: mergekit-yaml [OPTIONS] CONFIG_FILE OUT_PATH
mergekit-yaml {options} {config_file} {output_dir}

echo "Merge completed!"
echo "Output directory: {output_dir}"
"""
    return pbs_script, hostname

def _shorten_model_name(model_path: str) -> str:
    """Creates a shorter, more readable name from a model path."""
    name = Path(model_path).name
    
    # Rule for HF model names like 'pmahdavi/Llama-3.1-8B-math-reasoning'
    if 'Llama-3.1-8B-' in name:
        return name.split('Llama-3.1-8B-')[-1]

    # Rule for local checkpoints
    if name.startswith('checkpoint-'):
        parent_name = Path(model_path).parent.name
        if 'mixture_' in parent_name:
            return parent_name.split('mixture_')[-1].split('_full')[0]
        return parent_name

    # Fallback to the original name if no rules match
    return name

def generate_merge_name(config_file):
    """Generate an adaptive name for the merge job based on the config file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        merge_method = config.get('merge_method', 'unknown')
        
        # --- Model Names ---
        model_names = []
        if 'models' in config and isinstance(config['models'], list):
            for model_entry in config['models']:
                if 'model' in model_entry:
                    short_name = _shorten_model_name(model_entry['model'])
                    model_names.append(short_name)
        
        # --- Hyperparameters ---
        h_params = []
        config_params = config.get('parameters', {})
        
        # Method-specific hyperparameters
        if merge_method in ['dare_ties', 'dare_linear', 'ties', 'sce', 'breadcrumbs', 'breadcrumbs_ties', 'ffg']:
            density = config_params.get('density')
            if density is not None: h_params.append(f"d{density}")
        
        # OTA-specific hyperparameters
        if merge_method == 'ota':
            power = config_params.get('power')
            if power is not None: h_params.append(f"pow{power}")
            
            threshold = config_params.get('precond_threshold')
            if threshold is not None: h_params.append(f"precond-thresh{threshold:.0e}")

        # Common hyperparameters
        normalise = config_params.get('normalise')
        if normalise is not None and normalise != 'none':
            h_params.append(f"norm{normalise}")
            
        epsilon = config_params.get('epsilon')
        if epsilon is not None:
             h_params.append(f"eps{epsilon:.0e}")

        # --- Assemble Name ---
        components = [merge_method]
        
        if model_names:
            components.append("-".join(model_names))
            
        if h_params:
            components.append("_".join(h_params))

        merge_name = "_".join(components)
        
        # Sanitize and limit length
        merge_name = merge_name.replace('/', '-').replace(' ', '_')
        return merge_name[:150]

    except Exception as e:
        print(f"Warning: Could not parse config file or generate name: {e}", file=sys.stderr)
        return f'mergekit_fallback_{Path(config_file).stem}'

def generate_job_id():
    """Generate a unique job ID for the PBS script file name."""
    timestamp = time.strftime("%Y%m%d%H%M%S")
    random_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{random_id}"

def cleanup_temp_files(temp_script_path):
    """Clean up temporary script files after job submission."""
    try:
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
            print(f"Removed temporary PBS script: {temp_script_path}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}", file=sys.stderr)

def get_system_info():
    """Get current system information for dry run."""
    system_info = {}
    
    # Platform information
    system_info["os"] = platform.platform()
    system_info["hostname"] = socket.gethostname()
    
    # Python version
    system_info["python_version"] = platform.python_version()
    
    # Check for CUDA and torch
    try:
        import torch
        system_info["cuda_available"] = torch.cuda.is_available()
        system_info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"  # type: ignore[attr-defined]
        system_info["torch_version"] = torch.__version__
    except ImportError:
        system_info["cuda_available"] = "Unknown (torch not installed)"
        system_info["cuda_version"] = "Unknown"
        system_info["torch_version"] = "Not installed"
    
    # Check for mergekit
    try:
        import pkg_resources
        system_info["mergekit_version"] = pkg_resources.get_distribution("mergekit").version
    except:
        system_info["mergekit_version"] = "Unknown"
    
    # Hardware info
    try:
        # CPU info
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Model name' in line:
                    system_info["cpu_model"] = line.split(':')[1].strip()
                    break
        else:
            system_info["cpu_model"] = "Unknown"
            
        # GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            system_info["gpu_model"] = result.stdout.strip()
        else:
            system_info["gpu_model"] = "No GPU or nvidia-smi not available"
    except:
        system_info["cpu_model"] = "Could not determine"
        system_info["gpu_model"] = "Could not determine"
    
    return system_info

def print_dry_run_info(config_file, merge_method, output_dir, pbs_script, hostname, walltime, ngpus, ncpus, mem, job_name, merge_name):
    """Print information for a dry run."""
    print("\n" + "="*80)
    print(f"DRY RUN MODE - NO JOB WILL BE SUBMITTED")
    print("="*80 + "\n")
    
    # Print resource reservations directly from CLI arguments
    print(f"RESOURCE RESERVATION:")
    print(f"  - GPUs: {ngpus}")
    print(f"  - CPUs: {ncpus}")
    print(f"  - Memory: {mem}")
    print(f"  - Walltime: {walltime}")
    print(f"  - Queue: workq@{hostname}")
    print(f"  - Job Name: {job_name}")
    print()
    
    print(f"MERGE CONFIGURATION:")
    print(f"  - Merge Name: {merge_name}")
    print(f"  - Config File: {config_file}")
    print(f"  - Merge Method: {merge_method}")
    print(f"  - Output Directory: {output_dir}")
    print()
    
    print("PBS SCRIPT:")
    print("-" * 40)
    print(pbs_script)
    print("-" * 40)
    
    print("\nTo submit this job for real, run without the --dry-run flag.")
    print("="*80 + "\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run mergekit job with adaptive output directory')
    
    parser.add_argument('config_file', help='Path to the YAML config file')
    parser.add_argument('--output_dir', 
                        default='/scratch/pxm5426/runs/lora-exploration/merged_models',
                        help='Base output directory')
    parser.add_argument('--walltime', default='12:00:00', 
                        help='Walltime for the job (default: 12:00:00)')
    parser.add_argument('--ngpus', default='2', 
                        help='Number of GPUs for the job (default: 2)')
    parser.add_argument('--ncpus', default='16', 
                        help='Number of CPUs for the job (default: 16)')
    parser.add_argument('--mem', default='80g', 
                        help='Memory for the job (default: 80g)')
    parser.add_argument('--job_name', default='mergekit',
                        help='Name of the job for PBS (default: mergekit)')
    parser.add_argument('--keep_temp_files', action='store_true',
                        help='Keep temporary files after job submission')
    parser.add_argument('--no_allow_crimes', action='store_true',
                        help='Disable the --allow-crimes flag')
    parser.add_argument('--no_trust_remote_code', action='store_true',
                        help='Disable the --trust-remote-code flag')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show the PBS script and job details without submitting the job')
    
    args = parser.parse_args()
    
    # Read the config file to get the merge method
    try:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        merge_method = config.get('merge_method', 'unknown')
    except Exception as e:
        print(f"Error reading config file: {e}", file=sys.stderr)
        merge_method = "unknown"
    
    # Generate the merge name
    merge_name = generate_merge_name(args.config_file)
    
    # Create the full output directory path
    output_dir = os.path.join(args.output_dir, merge_name)
    
    # Make sure the directory for PBS output exists
    os.makedirs("pbs_results", exist_ok=True)
    
    # Generate a unique job ID for the PBS script file
    job_id = generate_job_id()
    
    # Create a temporary PBS script
    allow_crimes = not args.no_allow_crimes
    trust_remote_code = not args.no_trust_remote_code
    pbs_script, detected_hostname = create_pbs_script(
        args.config_file,
        merge_method,
        output_dir,
        args.walltime,
        args.ngpus,
        args.ncpus,
        args.mem,
        args.job_name,
        allow_crimes,
        trust_remote_code
    )
    
    temp_script_path = f"temp_mergekit_{job_id}.job"
    
    # In dry run mode, just print information without submitting
    if args.dry_run:
        print_dry_run_info(
            args.config_file,
            merge_method,
            output_dir,
            pbs_script,
            detected_hostname,
            args.walltime,
            args.ngpus,
            args.ncpus,
            args.mem,
            args.job_name,
            merge_name
        )
        return
    
    # Write the PBS script to a file
    with open(temp_script_path, 'w') as f:
        f.write(pbs_script)
    
    # Submit the job
    print(f"Submitting job '{args.job_name}' (Merge: {merge_name}) with config '{args.config_file}'...")
    print(f"Output directory: {output_dir}")
    print(f"Resource configuration: {args.ngpus} GPUs, {args.ncpus} CPUs, {args.mem} memory, {args.walltime} walltime")
    print(f"Target host: {detected_hostname}")
    
    result = subprocess.run(['qsub', temp_script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip()
        print(f"✓ Job submitted successfully: {job_id} (Merge name: {merge_name})")
    else:
        print(f"✗ Job submission failed: {result.stderr}")
    
    # Clean up temporary files
    if not args.keep_temp_files:
        cleanup_temp_files(temp_script_path)
    else:
        print(f"Keeping temporary files as requested:")
        print(f"  - PBS script: {temp_script_path}")

if __name__ == "__main__":
    main() 