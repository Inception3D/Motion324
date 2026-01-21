# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modifications Copyright (c) 2025 Hongyuan Chen.

from omegaconf import OmegaConf
import argparse
from easydict import EasyDict as edict
import re
import os
import datetime
import torch
import torch.distributed as dist
import numpy as np
import random
import yaml
import wandb
import shutil
import copy
from pathlib import Path
import time
import builtins

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print

#################Init Config  Begins#################
def process_overrides(overrides):
    """
    Handle space around "="
    """
    # First, join all items with spaces to create a single string
    combined = ' '.join(overrides)
    
    # Use regex to identify and fix patterns like 'param = value' to 'param=value'
    # This handles various spacing around the equals sign
    fixed_string = re.sub(r'(\S+)\s*=\s*(\S+)', r'\1=\2', combined)
    
    # Split the fixed string back into a list, preserving properly formatted args
    # We split on spaces that are not within a parameter=value pair
    processed = re.findall(r'[^\s=]+=\S+|\S+', fixed_string)
    
    return processed

def init_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*")  # Capture all "key=value" args
    args = parser.parse_args()

    # Load base config
    config = OmegaConf.load(args.config)

    # Parse CLI overrides using OmegaConf's native CLI parser
    processed_overrides = process_overrides(args.overrides)
    cli_overrides = OmegaConf.from_cli(processed_overrides)

    # Merge configs (with type-safe automatic conversion)
    config = OmegaConf.merge(config, cli_overrides)

    # Convert to EasyDict if needed
    config = OmegaConf.to_container(config, resolve=True)
    config = edict(config)
    return config

#################Init Config End#################


def init_distributed(seed=42):
    """
    Initialize distributed training environment and set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed for PyTorch, NumPy, and Python's random module.
                   Default is 42.
    
    Returns:
        edict: Dictionary with attribute access containing:
            - local_rank: GPU rank within the current node
            - global_rank: Global rank of the process
            - world_size: Total number of processes
            - device: The CUDA device assigned to this process
            - is_main_process: Flag to identify the main process
            - seed: The random seed used for this process
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        local_rank = global_rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)
        return edict({
            'local_rank': 0,
            'global_rank': 0,
            'world_size': 1,
            'device': torch.device('cuda'),
            'is_main_process': True, 
            'seed': seed,
            'is_distributed': False
        })
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,#
        rank=global_rank,#
        device_id=device,#
        timeout=datetime.timedelta(seconds=3600)
    )
    
    # Set random seeds
    # Each process gets a different seed derived from the base seed
    process_seed = seed + global_rank
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed) 
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    # Optional: For better performance
    torch.backends.cudnn.benchmark = True
    
    return edict({
        'local_rank': local_rank,
        'global_rank': global_rank,
        'world_size': world_size,
        'device': device,
        'is_main_process': global_rank == 0, 
        'seed': process_seed,
        'is_distributed': True
    })

def local_backup_src_code(
    src_dir,
    dst_dir,
    max_size_MB=20.0,
    extension_to_backup=(".py", ".yaml", ".sh", ".bash", ".json"),
    exclude_dirs=("wandb", ".git", "checkpoints", "experiments"),
    verbose=True,
):
    """
    Backup source code files with size limit check.
    
    Args:
        src_dir: Source directory to backup
        dst_dir: Destination directory for backups
        max_size_MB: Maximum total size allowed for backup in MB
        extension_to_backup: File extensions to include in backup
        exclude_dirs: Directories to exclude from backup
        verbose: Whether to print progress information
    
    Returns:
        tuple: (num_files_backed_up, total_size_in_bytes)
    
    Raises:
        ValueError: If total size exceeds max_size_MB
    """
    start_time = time.time()
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()
    
    # Convert to set for faster lookup
    extension_set = set(extension_to_backup)
    ignore_paths = {(src_path / d).resolve() for d in exclude_dirs}
    
    max_bytes = int(max_size_MB * 1024 * 1024)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")
    
    files = []
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(src_path):
        current_path = Path(dirpath).resolve()
        
        # Skip excluded directories
        if any(parent in ignore_paths for parent in current_path.parents) or current_path in ignore_paths:
            dirnames.clear()
            continue
        
        # Filter files by extension
        for filename in filenames:
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in extension_set:
                continue
                
            src_file = current_path / filename
            rel_path = current_path.relative_to(src_path)
            dst_file = dst_path / rel_path / filename
            
            try:
                file_size = src_file.stat().st_size
                total_size += file_size
                files.append((src_file, dst_file, file_size))
            except (FileNotFoundError, PermissionError) as e:
                if verbose:
                    print(f"Warning: Could not access {src_file}: {e}")
    
    if total_size > max_bytes:
        if verbose:
            print(f"Size limit exceeded: {total_size / (1024*1024):.2f} MB > {max_size_MB} MB")
            print("Largest files:")
            for src_file, _, size in sorted(files, key=lambda x: x[2], reverse=True)[:5]:
                print(f"{src_file}: {size / 1024:.1f} KB")
        raise ValueError(f"Size limit exceeded: {total_size / (1024*1024):.2f} MB > {max_size_MB} MB")
    
    if verbose:
        print(f"Backing up {len(files)} files ({total_size / (1024*1024):.2f} MB)")
    
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    successful_copies = 0
    for src_file, dst_file, _ in files:
        try:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            successful_copies += 1
        except Exception as e:
            if verbose:
                print(f"Error copying {src_file} to {dst_file}: {e}")
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Backup completed: {successful_copies}/{len(files)} files copied in {elapsed_time:.2f} seconds")
    
    return successful_copies, total_size
    


def init_wandb_and_backup(config):
    # API key validation
    assert os.path.exists(
        config.training.api_key_path
    ), f"API key file does not exist: {config.training.api_key_path}"
    api_keys = edict(yaml.safe_load(open(config.training.api_key_path, "r")))
    assert api_keys.wandb is not None, "Wandb API key not found in api key file"

    # WandB setup and login
    os.environ["WANDB_API_KEY"] = api_keys.wandb

    # WandB initialization
    config_copy = copy.deepcopy(config)
    wandb.init(
        project=config.training.wandb_project,
        name=config.training.wandb_exp_name,
        config=config_copy,
    ) 

    # Source code backup
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    trgt_dir = os.path.join(config.training.checkpoint_dir, "src", os.path.basename(cur_dir))
    os.makedirs(trgt_dir, exist_ok=True)
    extension_to_backup=(".py", ".yaml", ".sh") # ".bash", ".json"
    exclude_dirs=("wandb", ".git", "checkpoints", "experiments", ".history")
    local_backup_src_code(cur_dir, trgt_dir, extension_to_backup=extension_to_backup, exclude_dirs=exclude_dirs)

    # Save config file
    config_save_path = os.path.join(config.training.checkpoint_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(dict(config), f)

    wandb.run.log_code(
        trgt_dir,  
        include_fn=lambda path: path.endswith(extension_to_backup),
    )


