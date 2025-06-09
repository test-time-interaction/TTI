import os
import torch
import paramiko
import subprocess
import time
import json
import glob
import logging
from datetime import datetime
import threading
from tqdm import tqdm
import deepspeed


import os
import time
import torch
import logging
import deepspeed
from datetime import datetime
from tqdm import tqdm
import numpy as np

from tti.environment import batch_interact_environment
from tti.misc import colorful_print
from tti.models import GemmaVllmAgent

def worker_collect_loop(env,
                agent,prompt_processor,
                rollout_size: int = 32,
                batch_size: int = 2,
                grad_accum_steps: int = 1,
                capacity: int = 500000,
                epochs: int = 1,
                lm_lr: float = 1e-5,
                gamma: float = 0.9,
                use_wandb: bool = False,
                online: bool = True,
                eval_env = None,
                actor_epochs: int = 0,
                actor_trajectories: int = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                checkpoint_path: str = None,
                save_freq: int = 1,
                eval_freq: int = 10,
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                safe_batch_size: int = 4,
                eval_at_start: bool = False,
                algorithm: str = "filteredbc",
                worker_id: str = None,
                shared_model_path: str = None,
                policy_lm: str = None,
                train_tasks: str = '',
                test_tasks: str = '',
                evaluator_prompt_path: str = '',
                reset_server: bool = False,
                **kwargs):
    """
    Worker-specific collection loop that runs independently on worker nodes.
    
    This function runs on worker nodes to collect trajectories which are then saved
    with unique filenames based on worker ID. It does not perform any training.
    
    Args:
        env: Environment object for data collection
        agent: Agent for inference
        rollout_size: Number of trajectories to collect per worker
        ... (other parameters from onpolicy_train_loop)
    """
    if deepspeed.comm.get_rank() == 0:
        colorful_print(f"Starting worker collection loop", fg='green')
        
        # Initialize worker configuration
        worker_meta = worker_initialize({
            "worker_id": worker_id,
            "save_path": save_path,
            "shared_model_path": shared_model_path
        })
        
        # if shared_model_path does not exist, fall back to policy_lm
        if not os.path.exists(shared_model_path):
            shared_model_path = policy_lm
        
        # Prepare the agent for inference
        agent.enter_infer_mode(updated_model_path=shared_model_path)
        
        if reset_server and env:
            env.reset_server()
            
        # Log collection start
        colorful_print(f"Worker {worker_meta['worker_id']} collecting {rollout_size} trajectories", fg='cyan')
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Collect trajectories in one go using batch_interact_environment
        colorful_print(f"=====>Maximum Train-time Trajectory Length: {env.max_iter}", fg='green')
        collected_trajectories, _ = batch_interact_environment(
            agent=agent,
            prompt_processor = prompt_processor,
            env=env,
            num_trajectories=rollout_size,
            use_tqdm=True,
            decode_f=decode_f,
            gamma=gamma,
            safe_batch_size=safe_batch_size,
            iter=0
        )
            
        # Calculate total collection time
        collection_time = time.time() - start_time
        
        # Save the final results
        final_timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")
        final_path = worker_save_trajectories(collected_trajectories, worker_meta, final_timestamp)
        
        # Log collection completion
        colorful_print(f"Worker {worker_meta['worker_id']} completed collection of {len(collected_trajectories)} trajectories "
                      f"in {collection_time:.2f} seconds "
                      f"({len(collected_trajectories) / collection_time:.2f} trajectories/second)", fg='green')
        
        # Create a completion marker file
        completion_file = os.path.join(worker_meta["output_path"], f"{worker_meta['worker_id']}_COMPLETED")
        with open(completion_file, "w") as f:
            f.write(f"Completed at: {datetime.now().isoformat()}\n")
            f.write(f"Trajectories: {len(collected_trajectories)}\n")
            f.write(f"Collection time: {collection_time:.2f} seconds\n")
            f.write(f"Rate: {len(collected_trajectories) / collection_time:.2f} trajectories/second\n")
            f.write(f"Final save path: {final_path}\n")
        
        # Cleanup - reset environment if needed
        if env:
            try:
                env.close()
            except:
                pass
    
    # Synchronize all processes
    deepspeed.comm.barrier()
    
    colorful_print(f"Worker {worker_id} has finished collecting trajectories, errors after this can be safely ignored", fg='green')
    return

def worker_initialize(config):
    """
    Initialize a worker node with the proper environment setup.
    Called at the start of worker_collect_loop.
    
    Args:
        config: Configuration object with worker settings
    
    Returns:
        dict: Worker metadata
    """
    # Get worker ID from environment or config
    print(config)
    worker_id = config['worker_id']
    
    # Get output path from environment or config
    output_path = config['save_path']
    
    # Get shared model path from environment or config
    shared_model_path = config['shared_model_path']
    
    # Create a metadata object to identify this worker
    worker_metadata = {
        "worker_id": worker_id,
        "hostname": subprocess.check_output("hostname", shell=True).decode().strip(),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M_%S"),
        "output_path": output_path,
        "gpus": subprocess.check_output("nvidia-smi --list-gpus | wc -l", shell=True).decode().strip(),
        "shared_model_path": shared_model_path
    }
    
    # Create the output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Log worker initialization
    if deepspeed.comm.get_rank() == 0:
        logging.info(f"Worker initialized: {json.dumps(worker_metadata, indent=2)}")
        
        # Save metadata to a file
        metadata_file = os.path.join(output_path, "worker_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(worker_metadata, f, indent=2)
    
    return worker_metadata


def worker_save_trajectories(trajectories, metadata, timestamp=None):
    """
    Save trajectories collected by a worker with a unique name.
    
    Args:
        trajectories (list): List of trajectories to save
        metadata (dict): Worker metadata from worker_initialize
        timestamp (str, optional): Optional timestamp override
    
    Returns:
        str: Path to the saved trajectory file
    """
    if not trajectories:
        logging.warning("No trajectories to save")
        return None
    
    # Generate a timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M_%S")
    
    # Create a unique filename using worker ID and timestamp
    worker_id = metadata["worker_id"]
    output_path = metadata["output_path"]
    
    # Create the filename
    filename = f"{worker_id}_{timestamp}_trajectories.pt"
    filepath = os.path.join(output_path, filename)
    
    # Save the trajectories
    if deepspeed.comm.get_rank() == 0:
        # Add metadata to each trajectory for deduplication
        for traj in trajectories:
            if traj and len(traj) > 0:
                for step in traj:
                    if "metadata" not in step:
                        step["metadata"] = {"worker_id": worker_id, "timestamp": timestamp}
        
        # Save to disk
        torch.save(trajectories, filepath)
        logging.info(f"Saved {len(trajectories)} trajectories to {filepath}")
    
    return filepath
