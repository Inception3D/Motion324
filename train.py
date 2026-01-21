# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
# Modifications Copyright (c) 2025 Hongyuan Chen.

import importlib
import os
import sys
sys.setrecursionlimit(100000)

import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0, check_and_handle_global_nan_loss
from dataset.dyscene import collate_fn_with_topology

n_thread = 2
os.environ["MKL_NUM_THREADS"] = f"{n_thread}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_thread}" 
os.environ["OMP_NUM_THREADS"] = f"4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_thread}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_thread}"
os.environ["OMP_NUM_THREADS"] = f"{n_thread}"

# Load config and read(override) arguments from CLI
config = init_config()

ddp_info = init_distributed(seed=777)

if ddp_info.is_distributed:
    dist.barrier()
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
if ddp_info.is_distributed:
    dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}

# Load dataset, model, training settings and scheduler
dataset_name = config.training.get("dataset_name", "dataset.dyscene.Dyscene16k_Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config.training)
batch_size_per_gpu = config.training.batch_size_per_gpu

print_rank0(f"Training Dataset class_name: {class_name}")

if ddp_info.is_distributed:
    datasampler = DistributedSampler(dataset, shuffle=True)
else:
    datasampler = torch.utils.data.RandomSampler(dataset)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
    collate_fn=collate_fn_with_topology,
)

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
stop_steps = config.training.stop_steps * grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

module, class_name = config.model.class_name.rsplit(".", 1)
Motion324 = importlib.import_module(module).__dict__[class_name]
model = Motion324(config).to(ddp_info.device)

if ddp_info.is_distributed:
    model = DDP(model, device_ids=[ddp_info.local_rank])
else:
    model = model.to(ddp_info.device)

optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())

scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)

if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
)

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)

print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
print_rank0(config)

if ddp_info.is_distributed:
    dist.barrier()

start_train_step = cur_train_step
model.train()

dataloader_iter = iter(dataloader)
while cur_train_step <= stop_steps or cur_param_update_step <= total_param_update_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    try:
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        if ddp_info.is_distributed:
            datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}
    batch["cur_train_step"] = cur_train_step
    try:
        with torch.autocast(
            enabled=config.training.use_amp,
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.amp_dtype],
        ):
            ret_dict = model(batch)

        update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
        
        if not update_grads:
            if ddp_info.is_distributed:
                with model.no_sync(): # no sync grads for efficiency
                    scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
            else:
                scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
        else:
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
        
        cur_train_step += 1

        skip_optimizer_step = False
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        # Check gradient norm and update optimizer if everything is fine
        total_grad_norm = None
        if update_grads:
            scaler.unscale_(optimizer) 
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

            # visualize the grad norm of each layer of our model (FOR DEBUG)
            if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                grad_norms = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norms[name] = param.grad.detach().norm().item()
                for layer_name, grad_norm in grad_norms.items():
                    wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_param_update_step)

            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_param_update_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                if ddp_info.is_main_process:
                    if not skip_optimizer_step:
                        wandb.log({"grad_norm": total_grad_norm}, step=cur_param_update_step)

            if not skip_optimizer_step:
                scaler.step(optimizer)
                scaler.update()
                cur_param_update_step += 1

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    except RuntimeError as e:
        print_rank0(f"Step {cur_train_step}: RuntimeError during backward: {e}. Zeroing grads and skipping step.")
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        continue
    
    # log and save checkpoint
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            iter_time = time.time() - tic
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {iter_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            if config.training.grad_clip_norm > 0 and total_grad_norm is not None:
                print_str += f"Grad Norm: {total_grad_norm:.10f} | "
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 2000 + start_train_step
        ):
            iter_time = time.time() - tic
            log_dict = {
                "iter": cur_train_step, 
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": iter_time,
                "grad_norm": total_grad_norm if total_grad_norm is not None else 0.0,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            if not skip_optimizer_step and update_grads:
                wandb.log(
                    log_dict,
                    step=cur_param_update_step,
                )

        if (cur_param_update_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_param_update_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_param_update_step} to {os.path.abspath(ckpt_path)}")
    
if ddp_info.is_distributed:
    dist.barrier()
    dist.destroy_process_group()
