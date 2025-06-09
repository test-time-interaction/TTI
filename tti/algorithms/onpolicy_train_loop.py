from tti.environment import batch_interact_environment
import numpy as np
from tti.algorithms.filteredbc import BCTrainer
from tti.algorithms.base import BaseTrainer
from tti.misc import colorful_print
import wandb
import os
import torch
import deepspeed
import time
from tti.models import GemmaVllmAgent
from multiprocessing.pool import ThreadPool
from .utils import calculate_upperbound,\
    filter_trajectories, rollout_statistics_by_websites,\
    clean_trajectories
import concurrent.futures
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import psutil
import gc
import json

def onpolicy_train_loop(env,\
                agent, prompt_processor,\
                rollout_size: int = 50,
                batch_size: int = 1,
                grad_accum_steps: int = 2,
                capacity: int = 500000,
                epochs:int = 3, \
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                use_wandb: bool = False,
                online: bool = False,
                eval_env = None,
                actor_epochs: int = 3,
                actor_trajectories: int = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                checkpoint_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 10,
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                safe_batch_size: int = 4,
                eval_at_start: bool = False,
                algorithm: str = "filteredbc",
                parallel_option: str = "single",
                worker_ips: list = [],
                worker_username: str = "ubuntu",
                worker_run_path: str = "/home/ubuntu/project",
                host_run_path: str = "",
                remote_timeout: int = 10800,
                ssh_key_path: str = None,
                train_tasks: str = '',
                test_tasks: str = '',
                evaluator_prompt_path: str = '',
                eval_during_training: bool = True,
                **kwargs):
    if deepspeed.comm.get_rank() == 0:
        if env is not None:
            colorful_print("train env max iter: {}".format(env.max_iter), fg='green')
        if eval_env is not None:
            colorful_print("test env max iter: {}".format(eval_env.max_iter), fg='green')
        colorful_print(f"Using Algorithm {algorithm}", fg='green')
    train_trajectories = []
    # Load existing trajectories from save path
    if os.path.exists(os.path.join(save_path, 'train_trajectories.pt')):
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'), weights_only=False)
        if deepspeed.comm.get_rank() == 0:
            colorful_print(f"=====>Load existing trajectories for training from {save_path}", fg='green')
            
    evaluate_trajectories = []
    # Load existing evaluation trajectories from save path
    if os.path.exists(os.path.join(save_path, 'evaluate_trajectories.pt')):
        evaluate_trajectories = torch.load(os.path.join(save_path, 'evaluate_trajectories.pt'), weights_only=False)
        if deepspeed.comm.get_rank() == 0:
            colorful_print(f"=====>Load existing evaluation trajectories from {save_path}", fg='green')
    
    trainer = BCTrainer(agent=agent,\
                        lm_lr = lm_lr,\
                        batch_size = batch_size,\
                        gradient_accumulation_steps = grad_accum_steps,\
                        max_grad_norm=max_grad_norm,\
                        image_use_str=True)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Load GemmaVllmAgent from checkpoint if available
    if isinstance(agent, GemmaVllmAgent):
        if os.path.exists(os.path.join(save_path, "model.pt")):
            if deepspeed.comm.get_rank() == 0:
                colorful_print("=====>Loading from Gemma checkpoint", fg = 'green')
            updated_model_path = os.path.join(save_path, "model.pt")
            if os.path.exists(os.path.join(save_path, "ds_checkpoint")):
                if deepspeed.comm.get_rank() == 0:
                    colorful_print("=====>Loading from DeepSpeed checkpoint", fg = 'green')
                trainer_states_path = save_path
        else:
            if deepspeed.comm.get_rank() == 0:
                colorful_print(f"=====>Loading from RAW checkpoint rank {deepspeed.comm.get_rank()}", fg = 'green')
            updated_model_path = None
            trainer_states_path = None

    # Initial evaluation if requested
    agent.enter_train_mode(updated_model_path=updated_model_path)
    deepspeed.comm.barrier()
    
    if eval_at_start:
        info = {}
        if deepspeed.comm.get_rank() == 0:
            print("=====>Evaluating at start")
        
        agent.enter_infer_mode(updated_model_path=updated_model_path)
        if deepspeed.comm.get_rank() == 0:
            evaluate_trajectories = batch_interact_environment(agent = agent, prompt_processor = prompt_processor,\
                                    env = eval_env,\
                                    num_trajectories = len(eval_env.tasks),\
                                    use_tqdm = True,\
                                    decode_f = decode_f,\
                                    gamma = gamma,\
                                    safe_batch_size = safe_batch_size,\
                                    iter=0)
            
            info.update({"evaluate.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in evaluate_trajectories]),\
                         "evaluate.avg_steps": np.mean([len(d) for d in evaluate_trajectories]),\
                         "evaluate.num_trajectories": len(evaluate_trajectories),\
                "evaluate.timeouts": np.mean([len(d) >= eval_env.max_iter for d in evaluate_trajectories])})
            info.update(rollout_statistics_by_websites(evaluate_trajectories, eval_env, evaluate = True))
            torch.save(evaluate_trajectories, os.path.join(save_path, 'evaluate_trajectories.pt'))
            
            if use_wandb:
                wandb.log(info)
        deepspeed.comm.barrier()

    if deepspeed.comm.get_rank() == 0:
        colorful_print(f"The upperbound performance is {str(calculate_upperbound(train_trajectories))}", fg='green')

    if actor_epochs > 0:
        trainer.prepare(trainer_states_path=trainer_states_path)
    if deepspeed.comm.get_rank() == 0:
        print(f"RAM usage after trainer construction: {psutil.virtual_memory().percent}%")
    # Main training loop
    for epoch in range(epochs):
        
        gc.collect()
        torch.cuda.empty_cache() 
        info = {}
        if deepspeed.comm.get_rank() == 0:
            print(f"RAM usage before data collection: {psutil.virtual_memory().percent}%")
        
        if online:
            if deepspeed.comm.get_rank() == 0:
                colorful_print(f"=====>Online Epoch {epoch} rank {deepspeed.comm.get_rank()}", fg='green')
                colorful_print(f"=====>Maximum Train-time Trajectory Length: {env.max_iter}", fg='green')
            start_time = time.time()
        
            agent.enter_infer_mode(updated_model_path=updated_model_path)
            
            # Standard single-machine data collection
            if deepspeed.comm.get_rank() == 0:
                trajectories = batch_interact_environment(agent = agent,\
                                        env = env, prompt_processor = prompt_processor,\
                                        num_trajectories = rollout_size,\
                                        use_tqdm = True,\
                                        decode_f = decode_f,\
                                        gamma = gamma,\
                                        safe_batch_size = safe_batch_size,\
                                        iter=0)
                
                steps_pos_trajectories = []
                for d in trajectories:
                    if len(d) > 0 and d[0]["trajectory_reward"] > 0:
                        steps_pos_trajectories.append(len(d))
                info.update({"epoch": epoch,\
                        "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                        "rollout.avg_steps": np.mean([len(d) for d in trajectories]),\
                        "rollout.avg_steps_pos_trajectories": np.mean(steps_pos_trajectories),\
                        "rollout.num_trajectories": len(trajectories),\
                        "rollout.timeouts": np.mean([len(d) >= env.max_iter for d in trajectories]),\
                        "rollout.upperbound": calculate_upperbound(train_trajectories),\
                        "rollout.walltime": time.time()-start_time})
                if deepspeed.comm.get_rank() == 0:
                    colorful_print("Main thread finished collecting trajectories", fg='cyan')
                    colorful_print(f"Time taken to collect trajectories: {time.time() - start_time}", fg='cyan')
                info.update(rollout_statistics_by_websites(trajectories, env))
                train_trajectories += trajectories
                colorful_print(f"Saving {len(train_trajectories)} trajectories to {save_path}", fg='green')
                torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
                if use_wandb:
                    wandb.log(info)
            deepspeed.comm.barrier()
            
        if eval_during_training:
            info = {}
            if deepspeed.comm.get_rank() == 0:
                colorful_print(f"=====>Evaluating before Epoch {epoch}", fg='green')
            
            agent.enter_infer_mode(updated_model_path=updated_model_path)

            if deepspeed.comm.get_rank() == 0:
                trajectories = batch_interact_environment(agent = agent, prompt_processor = prompt_processor,\
                                        env = eval_env,\
                                        num_trajectories = len(eval_env.tasks),\
                                        use_tqdm = True,\
                                        decode_f = decode_f,\
                                        gamma = gamma,\
                                        safe_batch_size = safe_batch_size,\
                                        iter=0)
                
                evaluate_trajectories += trajectories
                
                steps_pos_trajectories = []
                for d in trajectories:
                    if len(d) > 0 and d[0]["trajectory_reward"] > 0:
                        steps_pos_trajectories.append(len(d))
                info.update({"evaluate.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "evaluate.avg_steps": np.mean([len(d) for d in trajectories]),\
                              "evaluate.num_trajectories": len(trajectories),\
                             "evaluate.avg_steps_pos_trajectories": np.mean(steps_pos_trajectories),\
                    "evaluate.timeouts": np.mean([len(d) >= eval_env.max_iter for d in trajectories])})
                info.update(rollout_statistics_by_websites(trajectories, eval_env, evaluate = True))
                colorful_print(f"Saving {len(evaluate_trajectories)} evaluation trajectories to {save_path}", fg='green')
                torch.save(evaluate_trajectories, os.path.join(save_path, f'evaluate_trajectories.pt'))
                if use_wandb:
                    wandb.log(info)
            deepspeed.comm.barrier()
            
        if actor_epochs > 0:
            train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'), weights_only=False)
            agent.enter_train_mode(updated_model_path=updated_model_path)
            deepspeed.comm.barrier()
            trainer.prepare(trainer_states_path=trainer_states_path)
            if deepspeed.comm.get_rank() == 0:
                print(f"RAM usage after data collection: {psutil.virtual_memory().percent}%")

            for aep in tqdm(range(actor_epochs), desc="Actor Training", disable=deepspeed.comm.get_rank() != 0):
                if algorithm == "filteredbc":
                    training_info = trainer.update(filter_trajectories(train_trajectories[-capacity:]), actor_trajectories=actor_trajectories, iter=epoch, sub_iter=aep)
                    info.update(training_info)
                elif algorithm == "sft":
                    training_info = trainer.update(clean_trajectories(train_trajectories[-capacity:]), actor_trajectories=actor_trajectories, iter=epoch, sub_iter=aep)
                    info.update(training_info)
                
            # Clean up training resources
            if deepspeed.comm.get_rank() == 0:
                print(f"RAM usage after training: {psutil.virtual_memory().percent}%")

            if (epoch + 1) % save_freq == 0 and not isinstance(trainer, BaseTrainer) and parallel_option in ["single", "host"] and actor_epochs > 0:
                trainer.save(os.path.join(save_path))
                deepspeed.comm.barrier()

            # Log to wandb only once, after all training subepochs are done and only from rank 0
            if use_wandb and deepspeed.comm.get_rank() == 0:
                # Use the global step from the trainer for consistent logging
                if hasattr(trainer, 'global_step'):
                    wandb.log(info, step=trainer.global_step)
                else:
                    wandb.log(info)

            if os.path.exists(os.path.join(save_path, "model.pt")):
                updated_model_path = os.path.join(save_path, "model.pt")

    trainer.cleanup()
