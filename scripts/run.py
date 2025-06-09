import sys
sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]
import torch
import transformers
from tti.models import GemmaVllmAgent
from tti.models.prompt_processor import PromptProcessor
from tti.algorithms import onpolicy_train_loop, worker_collect_loop
from tti.environment.webgym import BatchedWebEnv
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from datetime import timedelta
import json
from tti.environment.webgym.utils import replace_ec2_address
import deepspeed

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
transformers.logging.set_verbosity_error()

def load_task_file(assets_path, task_set, task_split):
    all_tasks = []
    with open(os.path.join(assets_path, f"{task_set}_{task_split}.txt")) as fb:
        for line in fb:
            all_tasks.append(line.strip())
    return all_tasks

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    # 1. Python random
    random.seed(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # 4. CUDNN (deterministic mode)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 5. Environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional but useful for reproducibility across workers in DDP
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # For CUDA 10.2+


@hydra.main(version_base=None, config_path="config/main", config_name="sft_llava")
def main(config: DictConfig):
    set_seed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    deepspeed.init_distributed(timeout=timedelta(minutes=3600))
    
    # Make sure batch_size and grad_accum_steps are in the config
    # If not specified, provide reasonable defaults
    if not hasattr(config, 'batch_size'):
        config.batch_size = 1
        print("Warning: batch_size not specified in config, defaulting to 1")
    
    if not hasattr(config, 'grad_accum_steps'):
        # For backward compatibility, use the existing field if available
        if hasattr(config, 'gradient_accumulation_steps'):
            config.grad_accum_steps = config.gradient_accumulation_steps
        else:
            config.grad_accum_steps = 1
            print("Warning: grad_accum_steps not specified in config, defaulting to 1")
    
    # Log the effective batch size
    effective_batch_size = config.batch_size * config.grad_accum_steps * deepspeed.comm.get_world_size()
    if deepspeed.comm.get_rank() == 0:
        print(f"Training with batch_size={config.batch_size}, grad_accum_steps={config.grad_accum_steps}")
        print(f"Effective global batch size: {effective_batch_size}")
    
    if config.agent_name == 'gemma-vllm':
        agent = GemmaVllmAgent(policy_lm = config.policy_lm, 
                               vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
                            config = config)
    else:
        raise NotImplementedError("Other agent model is not yet implemented")

    
    verbose = config.env_config.verbose if hasattr(config.env_config, "verbose") else False
    
    prompt_processor = PromptProcessor(config.prompt_path, config.evaluator_prompt_path, config.max_attached_imgs, verbose)
    
    tasks = []
    test_tasks = []
    if config.train_tasks is not None:
        with open(config.train_tasks, 'r', encoding='utf-8') as f:
            for line in f:
                    task_instance = json.loads(line) 
                    if task_instance['web_name'] == "Coursera":
                        task_instance['web'] = "https://www.coursera.org/"
                    if 'webarena' in config.train_tasks and hasattr(config.env_config, "webarena_subset") and config.env_config.webarena_subset:
                        if task_instance['web_name'] in config.env_config.webarena_subset.split("|"):
                            task_instance['web'] = replace_ec2_address(task_instance['web'], config.env_config.webarena_host)
                            tasks.append(task_instance)
                    else:
                        tasks.append(task_instance)
                        
    if config.test_tasks is not None:
        with open(config.test_tasks, 'r', encoding='utf-8') as f:
            for line in f:
                    task_instance = json.loads(line)
                        
                    if 'webarena' in config.test_tasks and hasattr(config.env_config, "webarena_subset") and config.env_config.webarena_subset:
                        if task_instance['web_name'] in config.env_config.webarena_subset.split("|"):
                            task_instance['web'] = replace_ec2_address(task_instance['web'], config.env_config.webarena_host)
                            test_tasks.append(task_instance)
                    else:
                        test_tasks.append(task_instance)
                        
    
    if deepspeed.comm.get_rank() == 0:
        print(f"Num training tasks: {len(tasks)} Num test tasks: {len(test_tasks)}")
    if config.train_tasks is not None and deepspeed.comm.get_rank() == 0:
        env = BatchedWebEnv(tasks = tasks,
                            download_dir=os.path.join(config.save_path, 'driver', 'download'),
                            output_dir=os.path.join(config.save_path, 'driver', 'output'),
                            env_config=config.env_config,
                            is_test=False)
    else:
        env = None

    if config.test_tasks is not None and deepspeed.comm.get_rank() == 0:
        test_env = BatchedWebEnv(tasks = test_tasks,
                                download_dir=os.path.join(config.save_path, 'test_driver', 'download'),
                                output_dir=os.path.join(config.save_path, 'test_driver', 'output'),
                                env_config=config.test_env_config,
                                is_test=True)
    else:
        test_env = None
        
    if config.use_wandb and deepspeed.comm.get_rank() == 0:
        run_id_path = os.path.join(config.save_path, "run_id.txt")
        run_id = None
        if os.path.isfile(run_id_path):
            try:
                with open(run_id_path, 'r') as f:
                    run_id = f.read().strip()
                print("[WANDB RESUME FROM PREVIOUS RUN]", run_id)
            except:
                pass
        
        wandb.login(key = config.wandb_key)
        if run_id is None:
            run = wandb.init(project = config.project_name, name = config.run_name, entity=config.entity_name, config = OmegaConf.to_container(config, resolve = True))
            with open(run_id_path, 'w') as f:
                f.write(run.id)
        else:
            run = wandb.init(project = config.project_name, id=run_id, resume="allow")
     
    # When passing to onpolicy_train_loop, make sure to include batch_size and grad_accum_steps explicitly
    if config.parallel_option in ["single", "host"]:
        onpolicy_train_loop(env = env,
                        agent = agent, prompt_processor = prompt_processor, eval_env = test_env,
                        **config)
    elif config.parallel_option == "worker":
        worker_collect_loop(env = env,
                        agent = agent, prompt_processor = prompt_processor, eval_env = test_env,
                        **config)
    
if __name__ == "__main__":
    main()
