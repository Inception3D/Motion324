# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modifications Copyright (c) 2025 Hongyuan Chen.

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import torch.distributed as dist
import os
from rich import print
import traceback
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import gc

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def create_optimizer(model, weight_decay, learning_rate, betas):
    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    decay_params, nodecay_params = [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,fused=True)
    
    if is_dist_avail_and_initialized():
        if dist.get_rank() == 0:
            def get_module_name(name):
                parts = name.split('.')
                if len(parts) > 2 and parts[0] == 'module':
                    return parts[1] + '.' + parts[2]
                return parts[0]
            print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight_decay: {weight_decay}, betas: {betas}')
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in optimized_param_dict.values())
            optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
            frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
            
            print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
            print(f'Optimized parameters: {optim_module_names}')
            print(f'Frozen parameters: {frozen_module_names}')
        
    return optimizer, optimized_param_dict, all_param_dict

def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler



def find_checkpoints(load_path):
    if os.path.isdir(load_path):
        ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
        ckpt_names = sorted(ckpt_names, key=lambda x: x)
        ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
    else:
        if load_path.endswith(".pt"):
            ckpt_paths = [load_path]
        else:
            ckpt_paths = []
    return ckpt_paths



def auto_resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state
):
    """
    Resume training from the latest checkpoint in the specified directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler: lr scheduler to be loaded
        reset_training_state: whether to reset the training state

    Returns:
        optimizer, lr_scheduler, forward_pass_step, param_update_step

    """
    forward_pass_step = 0
    param_update_step = 0
    all_ckpt_paths = find_checkpoints(load_path)
    if len(all_ckpt_paths) == 0:
        print_rank0(f"No checkpoint found in {load_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step
    try:
        ckpt_path = all_ckpt_paths[-1]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except:
        traceback.print_exc()
        print_rank0(f"Failed to load {ckpt_path}, we will start from scratch")
        return optimizer, lr_scheduler, forward_pass_step, param_update_step

    # Load model weights
    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    print_rank0(f"Loaded model from {os.path.abspath(ckpt_path)}, the status is {status}")

    if not reset_training_state:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            forward_pass_step = checkpoint["fwdbwd_pass_step"]
            param_update_step = checkpoint["param_update_step"]
            print_rank0(f"Resumed optimizer and lr_scheduler from {ckpt_path}")
        except:
            traceback.print_exc()
            print_rank0(f"Failed to load optimizer and lr_scheduler from {ckpt_path}")
    
    return optimizer, lr_scheduler, forward_pass_step, param_update_step


def check_and_handle_global_nan_loss(loss, ddp_info, cur_train_step, grad_accum_steps, total_train_steps, 
                               ret_dict, batch, lr_scheduler, optimizer):
    """
    Check if loss contains NaN/Inf values and handle cleanup if detected.
    
    Args:
        loss: The loss tensor to check
        ddp_info: Distributed training info object with is_distributed, device, etc.
        cur_train_step: Current training step
        grad_accum_steps: Gradient accumulation steps
        total_train_steps: Total training steps
        ret_dict: Return dictionary from model forward pass
        batch: Input batch
        lr_scheduler: Learning rate scheduler
        optimizer: Optimizer
    
    Returns:
        bool: True if NaN/Inf was detected and cleanup was performed, False otherwise
    """
    loss_is_nan = torch.isnan(loss) or torch.isinf(loss)
    if ddp_info.is_distributed:
        has_nan_tensor = torch.tensor(float(loss_is_nan), device=ddp_info.device)
        dist.all_reduce(has_nan_tensor, op=dist.ReduceOp.MAX)
        global_has_nan = has_nan_tensor.item() > 0.5
    else:
        global_has_nan = loss_is_nan

    if global_has_nan:
        print_rank0(f"Step {cur_train_step}: Detected NaN/Inf loss on at least one rank. Skipping backward and optimizer step for all ranks.")
        if (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps:
            print(f"Zeroing grads for step {cur_train_step}")
        
        def deep_detach(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach()
            elif isinstance(obj, dict):
                return {k: deep_detach(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(deep_detach(item) for item in obj)
            elif hasattr(obj, '__dict__'):
                for k, v in obj.__dict__.items():
                    setattr(obj, k, deep_detach(v))
                return obj
            else:
                return obj

        with torch.no_grad():
            ret_dict = deep_detach(ret_dict)

        del ret_dict
        del batch
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        
        if ddp_info.is_distributed:
            dist.barrier()
        
        return True
    
    return False


