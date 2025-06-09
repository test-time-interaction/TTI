import os
import json
import torch
import random
import deepspeed
import shutil
from tqdm import tqdm
from deepspeed.utils import safe_get_full_fp32_param
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Gemma3ForConditionalGeneration
from tti.data import TrajectoryStepDataset, custom_collate_fn
from tti.models import GemmaVllmAgent
from peft import PeftModel
from torch.utils.data._utils.collate import default_collate
import types
import time
import numpy as np
import gc
import wandb
from pathlib import Path
import re

def get_current_lr(optimizer):
     """Extract current learning rate from the optimizer."""
     if hasattr(optimizer, 'param_groups'):
         return optimizer.param_groups[0]['lr']
     # For optimizer wrapped by DeepSpeed
     elif hasattr(optimizer, 'optimizer'):
         return optimizer.optimizer.param_groups[0]['lr']
     # For deepspeed engine
     elif hasattr(optimizer, '_optimizer'):
         return optimizer._optimizer.param_groups[0]['lr']
     else:
         None

def dict_mean(dict_list):
    mean_dict = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class BCTrainer():
    def __init__(self, agent,
                    lm_lr: float = 1e-5,
                    batch_size: int = 4,
                    gradient_accumulation_steps: int = 1,  # New parameter
                    max_grad_norm: float = 1.0,
                    image_use_str: bool = False):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.lm_lr = lm_lr
        self.agent = agent
        self.model_engine = None
        self.batch_size = batch_size  # Use the provided batch size
        self.gradient_accumulation_steps = gradient_accumulation_steps  # Store gradient accumulation steps
        self.max_grad_norm = max_grad_norm
        self.image_use_str = image_use_str
        self.loss_log = []
        self.step_times = []
        self.is_main_process = deepspeed.comm.get_rank() == 0
        
        # Initialize global step counter - this will be persisted across training runs
        self.global_step = 0
        self.last_global_step = 0
        
        # Track micro steps for manual gradient accumulation
        self.micro_step = 0
        
        # Initialize wandb on main process only to prevent duplicate logging
        if self.is_main_process and wandb.run is not None:
            # Check if we can retrieve the global step from wandb
            if wandb.run.summary.get("global_step") is not None:
                self.global_step = wandb.run.summary.get("global_step")
                print(f"Resuming from global step {self.global_step} from wandb")
            
            # Store global step in wandb summary for persistence
            wandb.run.summary["global_step"] = self.global_step
    
    def estimate_memory_requirements(self):
        """
        Estimates memory requirements based on batch size and model size.
        Provides warnings if likely to encounter OOM errors.
        """
        if not self.is_main_process:
            return
            
        try:
            # Get available GPU memory in GB
            free_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Rough estimate - actual requirements vary by model architecture
            param_count = sum(p.numel() for p in self.agent.train_model.parameters()) / 1e9
            
            # Very rough memory estimation in GB - adjust based on your specific models
            # Formula: Parameters in BF16 (2 bytes) + optimizer states + activations + buffer
            estimated_memory = param_count * 2  # Parameters in BF16
            estimated_memory += param_count * 2  # Optimizer states (Adam has 2 states)
            estimated_memory += param_count * 0.5 * self.batch_size  # Activations (very rough)
            estimated_memory += 2  # Buffer
            
            # Apply crude scaling with ZeRO stage and offloading
            # ZeRO-3 with offloading significantly reduces memory requirements
            estimated_memory = estimated_memory / deepspeed.comm.get_world_size()
            estimated_memory = estimated_memory * 0.6  # Assume ZeRO-3 with offloading reduces memory by ~40%
            
            print(f"\n===== MEMORY REQUIREMENT ESTIMATE =====")
            print(f"Model parameters: {param_count:.2f} billion")
            print(f"Estimated memory per GPU: {estimated_memory:.2f} GB")
            print(f"Available GPU memory: {free_memory:.2f} GB")
            
            # Warning if close to limits
            if estimated_memory > free_memory * 0.9:
                print("\n⚠️ WARNING: You're likely to encounter CUDA OOM errors with current settings!")
                print(f"Consider reducing batch_size (currently {self.batch_size}) or")
                print(f"increasing gradient_accumulation_steps (currently {self.gradient_accumulation_steps})")
                print(f"to reduce memory usage. Effective batch size will remain the same.")
                print("Alternatively, you can enable further parameter offloading in DeepSpeed config.\n")
            elif estimated_memory > free_memory * 0.7:
                print("\n⚠️ CAUTION: You're using a significant portion of available GPU memory.")
                print("Keep an eye out for potential CUDA OOM errors during training.\n")
            else:
                print("\n✓ Your memory settings look reasonable for the available GPU memory.\n")
        except Exception as e:
            print(f"Error estimating memory requirements: {e}")
            print("Continuing with current settings, but be prepared for potential OOM errors.")
    
    def reduce_batch_size(self, factor=2):
        """
        Reduces batch size and adjusts gradient accumulation to maintain same effective batch size.
        Used for recovery from OOM errors.
        
        Args:
            factor: Factor by which to reduce batch size (and increase grad accum steps)
        """
        new_batch_size = max(1, self.batch_size // factor)
        new_grad_accum = self.gradient_accumulation_steps * (self.batch_size // new_batch_size)
        
        if self.is_main_process:
            print(f"\n===== ADJUSTING BATCH SETTINGS DUE TO OOM =====")
            print(f"Old settings: batch_size={self.batch_size}, grad_accum_steps={self.gradient_accumulation_steps}")
            print(f"New settings: batch_size={new_batch_size}, grad_accum_steps={new_grad_accum}")
            print(f"Effective batch size remains: {new_batch_size * new_grad_accum * deepspeed.comm.get_world_size()}")
        
        self.batch_size = new_batch_size
        self.gradient_accumulation_steps = new_grad_accum
        
        # Return True if we were able to reduce further, False if we hit the minimum
        return new_batch_size < self.batch_size

    def prepare(self, trainer_states_path):
        """
        Prepare the trainer for training, optionally loading from a checkpoint
        
        Args:
            trainer_states_path: Path to directory containing saved checkpoints
        """
        # Estimate memory requirements at the start
        if isinstance(self.agent, GemmaVllmAgent):
            if self.model_engine:
                self.model_engine.train()
                self.model_engine.gradient_checkpointing_enable()
            
                # Reset micro_step counter but retain global_step for consistent logging
                self.micro_step = 0
                self.optimizer.zero_grad()
                return
            
            self.estimate_memory_requirements()
            
            if self.is_main_process:
                print(f"Loading model and initializing DeepSpeed ZeRO-3 with batch size {self.batch_size} and gradient accumulation {self.gradient_accumulation_steps}")
            
            # Set training mode and requires_grad
            self.agent.train_model.train()
            
            # Enable gradients for all parameters
            for name, param in self.agent.train_model.named_parameters():
                if "vision_tower" in name: # or "multi_modal_projector" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            
            # Create parameter groups for optimizer
            model_params = [
                {"params": [p for p in self.agent.train_model.parameters() if p.requires_grad], 
                "lr": self.lm_lr}
            ]
            
            # Configure DeepSpeed for ZeRO-3
            ds_config = {
                "bf16": {
                    "enabled": True
                },
                "train_micro_batch_size_per_gpu": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "train_batch_size": self.batch_size * self.gradient_accumulation_steps * deepspeed.comm.get_world_size(),
                
                "zero_optimization": {
                    "stage": 3,
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e7,
                    "stage3_prefetch_bucket_size": 5e7,
                    "stage3_param_persistence_threshold": 1e5,
                    
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    
                    "round_robin_gradients": True,
                    "stage3_gather_16bit_weights_on_model_save": True
                },
                
                "activation_checkpointing": {
                    "partition_activations": True,
                    "cpu_checkpointing": True,
                    "contiguous_memory_optimization": True,
                    "number_checkpoints": 2,
                    "synchronize_checkpoint_boundary": True,
                    "profile": False
                },
                
                "autotuning": {
                    "enabled": False,
                },
                
                "gradient_clipping": self.max_grad_norm,
                "steps_per_print": 10,
                
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.lm_lr,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0.01
                    }
                },

                "scheduler": {
                    "type": "WarmupCosineLR", # WarmupCosineLR or WarmupLR
                    "params": {
                        "total_num_steps": 2000,
                        "warmup_num_steps": 20
                    }
                },
                
                "flops_profiler": {
                    "enabled": False,
                    "profile_step": -1,
                    "module_depth": -1,
                    "top_modules": 0,
                    "detailed": False,
                },
                
                "wall_clock_breakdown": False
            }

            # Set environment variables
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            os.environ["DISABLE_DEEPSPEED_FLOPS_PROFILER"] = "1"
            
            # Check if we have a checkpoint to load
            checkpoint_dir = None
            if trainer_states_path is not None:
                # Look for DeepSpeed checkpoint first
                ds_checkpoint_path = os.path.join(trainer_states_path, "ds_checkpoint")
                if os.path.exists(ds_checkpoint_path):
                    checkpoint_dir = ds_checkpoint_path
                    if self.is_main_process:
                        print(f"Found DeepSpeed checkpoint at {ds_checkpoint_path}, will load after initialization")
            
            # Initialize DeepSpeed
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.agent.train_model,
                model_parameters=model_params,
                config=ds_config)
            
            # Load checkpoint after initialization if available
            if checkpoint_dir is not None:
                if self.is_main_process:
                    print(f"Loading DeepSpeed checkpoint from {checkpoint_dir}")
                try:
                    _, client_state = self.model_engine.load_checkpoint(checkpoint_dir)
                    if self.is_main_process:
                        if client_state:
                            # Restore the global step if it was saved in the client state
                            if "global_step" in client_state:
                                previous_global_step = client_state.get("global_step", 0)
                                self.global_step = previous_global_step
                                print(f"Restored global step: {self.global_step} from checkpoint")
                            print(f"Successfully loaded checkpoint (epoch: {client_state.get('epoch', 'unknown')})")
                        else:
                            print("Successfully loaded checkpoint")
                except Exception as e:
                    if self.is_main_process:
                        print(f"Error loading checkpoint: {e}")
                        print("Continuing with fresh initialization")
            
            # Monkey patch the step function to avoid flops profiler
            original_step = self.model_engine.step
            
            def safe_step(self, *args, **kwargs):
                try:
                    return original_step(*args, **kwargs)
                except AttributeError as e:
                    if '__flops__' in str(e):
                        if self.optimizer is not None:
                            self.optimizer.step()
                            
                        if self.zero_optimization():
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            self.optimizer.zero_grad()
                    else:
                        raise
            
            # Apply the monkey patch
            self.model_engine.step = types.MethodType(safe_step, self.model_engine)
            
            # Enable gradient checkpointing
            self.model_engine.gradient_checkpointing_enable()
            
            # Reset micro_step counter but preserve global_step for consistent logging
            self.micro_step = 0
        
    def cleanup(self):
        """Clean up DeepSpeed resources before mode transitions"""
        if hasattr(self, "model_engine") and self.model_engine is not None:
            
            try:
                if hasattr(self.model_engine.optimizer, "destroy"):
                    self.model_engine.optimizer.destroy()
                
                try:
                    if hasattr(self.model_engine, "optimizer") and self.model_engine.optimizer is not None:

                        if hasattr(self.model_engine.optimizer, 'param_groups'):
                            for group in self.model_engine.optimizer.param_groups:
                                group['params'].clear()
                    
                        if hasattr(self.model_engine.optimizer, 'state'):
                            self.model_engine.optimizer.state.clear()
                    
                        if hasattr(self.model_engine.optimizer, 'fp32_groups'):
                            self.model_engine.optimizer.fp32_groups.clear()
                    
                        if hasattr(self.model_engine.optimizer, 'partition_gradients'):
                            self.model_engine.optimizer.partition_gradients.clear()
                    
                        if hasattr(self.model_engine.optimizer, 'cpu_offload'):
                            if hasattr(self.model_engine.optimizer.cpu_offload, 'buffer'):
                                self.model_engine.optimizer.cpu_offload.buffer = None
                        
                        # Set to None before deleting
                        optimizer = self.model_engine.optimizer
                        self.model_engine.optimizer = None
                        del optimizer

                except Exception as e:
                    print(f"Error engine opt clean {e}")

                try:
                    if hasattr(self, "optimizer") and self.optimizer is not None:
                        # Clear optimizer state dict explicitly
                        if hasattr(self.optimizer, 'param_groups'):
                            for group in self.optimizer.param_groups:
                                group['params'].clear()
                    
                        if hasattr(self.optimizer, 'state'):
                            self.optimizer.state.clear()
                    
                        if hasattr(self.optimizer, 'fp32_groups'):
                            self.optimizer.fp32_groups.clear()
                    
                        if hasattr(self.optimizer, 'partition_gradients'):
                            self.optimizer.partition_gradients.clear()
                    
                        if hasattr(self.optimizer, 'cpu_offload'):
                            if hasattr(self.optimizer.cpu_offload, 'buffer'):
                                self.optimizer.cpu_offload.buffer = None
                        
                        # Set to None before deleting
                        optimizer = self.optimizer
                        self.optimizer = None
                        del optimizer

                except Exception as e:
                    print(f"Error self opt clean {e}")
                    
                # Clean up optimizer references
                if hasattr(self.model_engine, "optimizer") and self.model_engine.optimizer is not None:
                    for param_group in self.model_engine.optimizer.param_groups:
                        for param in param_group["params"]:
                            if hasattr(param, "ds_tensor"):
                                del param.ds_tensor
                    self.model_engine.optimizer = None

                if hasattr(self, "optimizer") and self.optimizer is not None:
                    for param_group in self.optimizer.param_groups:
                        for param in param_group["params"]:
                            if hasattr(param, "ds_tensor"):
                                del param.ds_tensor
                    self.optimizer = None
                    
                # Remove engine reference
                if hasattr(self.agent, "model") and self.agent.train_model is not None:
                    if hasattr(self.agent.train_model, "module"):
                        del self.agent.train_model.module
                    del self.agent.train_model
                    self.agent.train_model = None

                for attr in [
                    'param_shapes',
                    'allgather_bucket',
                    'fp32_partitioned_groups',
                    'partitioned_optim_state_dict',
                    'param_names',
                    'ds_tensor',
                ]:
                    if hasattr(self.model_engine, attr):
                        delattr(self.model_engine, attr)
                             
                if hasattr(self.model_engine, "activation_checkpointing") and hasattr(self.model_engine.activation_checkpointing, "checkpointed_layers"):
                    self.model_engine.activation_checkpointing.checkpointed_layers.clear()

                for mod in self.model_engine.module.modules():
                    if hasattr(mod, "activation_checkpointing"):
                        mod.activation_checkpointing = None
                # Destroy the engine
                self.model_engine.destroy()
                engine = self.model_engine
                self.model_engine = None
                del engine
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()

                print("DeepSpeed resources cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up DeepSpeed resources: {e}")

    def actor_loss(self, message, action, **kwargs):
        """Compute actor loss with manual gradient accumulation for ZeRO-3."""
        # Zero gradients at the start of each accumulation cycle
        if self.micro_step == 0:
            self.model_engine.zero_grad()
        
        # try:
        log_probs = self.agent.get_log_prob(message, action)
        loss = -log_probs.mean()  # Always take mean for batched input

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        self.model_engine.backward(scaled_loss)
        
        # Get loss value for logging
        loss_value = scaled_loss.detach().cpu().item() * self.gradient_accumulation_steps
        
        # Increment micro step counter
        self.micro_step += 1
        
        # Only step after accumulation is complete
        need_step = self.micro_step >= self.gradient_accumulation_steps
        if need_step:
            # Perform optimizer step
            self.model_engine.step()
            # Reset counter
            self.micro_step = 0
        
        # Log only from main process
        if self.is_main_process:
            cur_lr = self.model_engine.lr_scheduler.get_last_lr()
            
            # Only log at intervals or when stepping to avoid step inconsistencies
            if self.global_step % 10 == 0 or need_step:
                print(f"\tStep: {self.global_step} | Loss: {loss_value:.6f} | LR: {cur_lr}")
                
                # Log to wandb if available - IMPORTANT: increment global_step only after logging
                if wandb.run is not None:
                    wandb.log({
                        "loss": loss_value,
                        "learning_rate": cur_lr
                    }, step=self.global_step)
        
        # Track global step - increment after logging to ensure consistent step counting
        self.global_step += 1
        return {"bc.loss": loss_value}

    def actor_validate(self, observation, action, **kwargs):
        with torch.no_grad():
            try:
                log_probs = self.agent.get_log_prob(observation, action)
                
                # Handle both batch_size=1 and batch_size>1 cases
                if self.batch_size == 1:
                    loss = -log_probs
                else:
                    loss = -log_probs.mean()
                
                loss_value = loss.detach().cpu().item()
                
                return {"validate.bc.loss": loss_value}
            except Exception as e:
                if self.is_main_process:
                    print(f"Error in validation: {e}")
                torch.cuda.empty_cache()
                return {"validate.bc.loss": 0.0}

    def update(self, trajectories, actor_trajectories, iter, sub_iter):
        # Reset tracking variables at the start of each update
        epoch_start_time = time.time()
        
        # Add explicit barrier for synchronization
        deepspeed.comm.barrier()
        
        self.agent.train_model.train()
        random.seed(iter)

        # Sample trajectories with weighted probability based on recency
        if actor_trajectories and actor_trajectories < len(trajectories):
            indices = np.arange(len(trajectories))
            recency_weights = indices + 1
            recency_bias_power = 1
            recency_weights = recency_weights ** recency_bias_power
            recency_weights = recency_weights / recency_weights.sum()
            
            # Calculate class weights based on inverse frequency
            web_names = {}
            for trajectory in trajectories:
                if len(trajectory) == 0:
                    continue
                if not trajectory[0]["observation"]["web_name"] in web_names:
                    web_names[trajectory[0]["observation"]["web_name"]] = 0
                web_names[trajectory[0]["observation"]["web_name"]] += 1
            
            # Calculate inverse frequency for each class
            total_samples = sum(web_names.values())
            class_weights_dict = {name: total_samples / (count * len(web_names)) for name, count in web_names.items()}
            
            # Apply class weights to trajectories
            class_weights = np.ones(len(trajectories))
            for i, trajectory in enumerate(trajectories):
                if len(trajectory) > 0:
                    web_name = trajectory[0]["observation"]["web_name"]
                    class_weights[i] = class_weights_dict[web_name]
            
            # Normalize class weights
            class_weights = class_weights / class_weights.sum()
            
            # Combine recency and class weights (you can adjust the relative importance)
            alpha = 0.5  # Adjust this value to control the balance between recency and class weights
            combined_weights = alpha * recency_weights + (1 - alpha) * class_weights
            
            # Normalize the combined weights
            recency_weights = combined_weights / combined_weights.sum()
            
            # Sample without replacement using the weights
            sampled_indices = np.random.choice(
                len(trajectories), 
                size=actor_trajectories, 
                replace=False, 
                p=recency_weights
            )
            
            # Get the sampled trajectories
            sampled_trajectories = [trajectories[i] for i in sampled_indices]
            trajectories = sampled_trajectories
            web_names = {}
            for trajectory in trajectories:
                if len(trajectory) == 0:
                    continue
                if not trajectory[0]["observation"]["web_name"] in web_names:
                    web_names[trajectory[0]["observation"]["web_name"]] = 0
                web_names[trajectory[0]["observation"]["web_name"]] += 1
            print(web_names)
        
        dataset = TrajectoryStepDataset(trajectories)
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset)
        
        # Create the DataLoader with the custom collate function.
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,       # This sampler is None if not in distributed mode.
            collate_fn=custom_collate_fn,
            shuffle=(sampler is None)  # Only shuffle if not using a distributed sampler.
        )
        
        # Reset micro_step counter at the beginning of each update
        self.micro_step = 0
        
        current_lr = get_current_lr(self.optimizer)
        
        # Log epoch start - only from main process
        if self.is_main_process:
            print(f"\n===== Actor Epoch {iter} Subepoch {sub_iter} Training on {len(dataset)} samples =====")
            print(f"Batch size: {self.batch_size}, Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps * deepspeed.comm.get_world_size()} (global)")
            print(f"Starting learning rate: {current_lr}")
            print(f"Current global step: {self.global_step}")

        # Training loop code
        info = {}
        info_list = []
        
        # Use tqdm only on the main process
        iterator = tqdm(dataloader, disable=not self.is_main_process)
        for step, sample in enumerate(iterator):
            batch_info = self.actor_loss(**sample)
            if self.micro_step != 0:
                self.model_engine.step()
                self.micro_step = 0
            if batch_info:
                info_list.append(batch_info)
            
            # Update tqdm with current loss and micro step info
            if self.is_main_process and 'bc.loss' in batch_info:
                iterator.set_postfix({
                    "loss": f"{batch_info['bc.loss']:.6f}", 
                    "micro_step": f"{self.micro_step}/{self.gradient_accumulation_steps}",
                    "global_step": self.global_step
                })
            
            # Memory cleanup
            torch.cuda.empty_cache()
        
        # Ensure all processes are synchronized before returning
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        deepspeed.comm.barrier()
        
        # Make sure micro_step is reset for the next update
        self.micro_step = 0
        
        # Log epoch summary - only from main process
        if self.is_main_process and wandb.run is not None:
            epoch_info = dict_mean(info_list) if info_list else {"bc.loss": 0.0}
            wandb.log({
                f"epoch_{iter}_summary/loss": epoch_info.get("bc.loss", 0.0),
                f"epoch_{iter}_summary/learning_rate": current_lr,
                "epoch": iter,
                "subepoch": sub_iter
            }, step=self.global_step)
        
        return dict_mean(info_list) if info_list else {"bc.loss": 0.0}

    def rotate_checkpoints(self, output_dir, prefix="global_step", max_checkpoints=1):
        """
        Keeps only the last `max_checkpoints` in `output_dir` with names matching `prefix`.
        """
        if max_checkpoints < 0:
            return
        
        ckpt_dir = Path(output_dir)
        pattern = re.compile(f"{prefix}[0-9]+")
    
        # Find matching checkpoint directories
        checkpoint_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and pattern.fullmatch(d.name)]
        
        # Sort by step number
        checkpoint_dirs = sorted(
            checkpoint_dirs,
            key=lambda x: int(x.name.replace(prefix, ""))
        )
        # Remove older ones
        if len(checkpoint_dirs) > max_checkpoints:
            to_delete = checkpoint_dirs[:-max_checkpoints]
            for d in to_delete:
                print(f"Removing old checkpoint: {d}")
                shutil.rmtree(d, ignore_errors=True)
                
    def save(self, path):
        """
        Save model in both DeepSpeed checkpoint format (for training) and
        HuggingFace format (for vLLM inference).
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        if self.is_main_process:
            print(f"\n===== Saving model to {path} =====")
        deepspeed.comm.barrier()
        
        # 1. First, save the complete DeepSpeed checkpoint with optimizer states
        ds_checkpoint_path = os.path.join(path, "ds_checkpoint")
        if self.is_main_process:
            print(f"Saving DeepSpeed checkpoint to {ds_checkpoint_path}")
        
        # Create a client state to save additional metadata
        client_state = {
            "epoch": iter,
            "global_step": self.global_step,
            "timestamp": time.time()
        }
        
        # The save_checkpoint method saves model weights, optimizer states, 
        # lr scheduler states, and other training state, including our client_state
        self.model_engine.save_checkpoint(ds_checkpoint_path, client_state=client_state)

        if self.is_main_process:
            self.rotate_checkpoints(ds_checkpoint_path, prefix="global_step", max_checkpoints=1)
            
            # Also update wandb if available
            if wandb.run is not None:
                wandb.run.summary["global_step"] = self.global_step
        
        # 2. Now save the HuggingFace format model for vLLM inference
        hf_model_path = os.path.join(path, "model.pt")
        
        # Only rank 0 proceeds with HF model preparation
        if self.is_main_process:
            # Remove existing directory if present
            if os.path.exists(hf_model_path):
                print(f"Removing existing model directory at {hf_model_path}")
                try:
                    shutil.copytree(hf_model_path, os.path.join(path, f"model_{self.global_step}.pt"))
                except:
                    pass
                shutil.rmtree(hf_model_path)
                # make dir
                os.makedirs(hf_model_path, exist_ok=True)
                    
            self.agent.processor.save_pretrained(hf_model_path)
            self.agent.tokenizer.save_pretrained(hf_model_path)
            # save the model config
            self.agent.train_model.config.save_pretrained(hf_model_path)
        self.last_global_step = self.global_step
        
        # Save the 16-bit model for vLLM inference
        self.model_engine.save_16bit_model(hf_model_path, "pytorch_model.bin")
        
        if self.is_main_process:
            print("✓ Model saved successfully in both formats:")
            print(f"  - DeepSpeed checkpoint (with optimizer states): {ds_checkpoint_path}")
            print(f"  - HuggingFace model (for vLLM inference): {hf_model_path}")
        
        deepspeed.comm.barrier()