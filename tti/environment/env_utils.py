import torch
from tqdm import tqdm
import numpy as np
import deepspeed
from tti.misc import colorful_print
from copy import deepcopy
import time
from collections import defaultdict
from tti.environment.webgym.utils_eval import webarena_batch_eval
import math
import re
import logging
import gc

def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})
    return trajectory

def find_most_common_action(action_keys, infos):
    """
    Find the index of the most commonly occurring (action_key, info) pair.
    
    Args:
        action_keys: List of action key strings (may contain None values)
        infos: List of info dictionaries (may contain None values)
        
    Returns:
        tuple: (index, count) of the most frequent action
    """    
    # Create a dictionary to count occurrences of each action-info pair
    action_counts = {}
    
    # Track which indices correspond to which action-info pairs
    action_indices = {}
    
    for i in range(len(action_keys)):
        action_key = action_keys[i]
        info = infos[i]
        
        # Skip None values
        if action_key is None and info is None:
            continue
        
        # Convert the info dictionary to a hashable format (tuple of sorted items)
        if isinstance(info, dict):
            hashable_info = tuple(sorted((k, v) for k, v in info.items()))
        else:
            # If info is already a simple type like a list
            hashable_info = tuple(info) if isinstance(info, list) else info
        
        # Create a composite key for the action-info pair
        composite_key = (action_key, hashable_info)
        
        # Increment the count for this action-info pair
        action_counts[composite_key] = action_counts.get(composite_key, 0) + 1
        
        # Store the first index where this combination appears
        if composite_key not in action_indices:
            action_indices[composite_key] = i
    
    # If no valid actions were found
    if not action_counts:
        return 0
    
    # Find the most common action-info pair
    most_common_action = max(action_counts.items(), key=lambda x: x[1])
    most_common_key = most_common_action[0]
    most_common_count = most_common_action[1]
    
    # Return the index of the most common action-info pair and its count
    return action_indices[most_common_key]
    
# deals with the case when the environment is done
def safe_batch_get_action(agent, prompt_processor, batch_obs, batch_done, batch_msg, batch_past_actions, steps, safe_batch_size = 4, n = 1, best_of_n=True):
    new_obs_idxs = []
    new_obs = []
    new_input_msg = []
    new_past_act = []
    batch_action_dict = [None] * len(batch_obs)
    
    for i, done in enumerate(batch_done):
        if not done and batch_obs[i] is not None:
            new_obs.append(batch_obs[i])
            new_obs_idxs.append(i)
            new_input_msg.append(batch_msg[i])
            new_past_act.append(batch_past_actions[i])

    if len(new_obs) > 0:
        new_input_msg = prompt_processor.process_batch_observation(new_obs, new_input_msg, steps)

        for i in range(0, len(new_obs), safe_batch_size):
            if n == 1:
                response,_ = agent.get_action(new_input_msg[i:i+safe_batch_size])
                action_keys, infos, messages = prompt_processor.process_batch_response(response, new_obs[i:i+safe_batch_size])
                
                for j, idx in enumerate(new_obs_idxs[i:i+safe_batch_size]):
                    batch_action_dict[idx] = {"action_key": action_keys[j], "info": infos[j]}
                    batch_msg[idx] = new_input_msg[j]
                    batch_msg[idx].append(messages[j])
            else:
                response_dict = defaultdict(list)
                prob_dict = defaultdict(list)
                bsz = len(new_input_msg[i:i+safe_batch_size])
                for sample_round in range(n):
                    response, prob = agent.get_action(new_input_msg[i:i+bsz])
                    
                    for batch_idx in range(bsz):
                        response_dict[batch_idx].append(response[batch_idx])
                        prob_dict[batch_idx].append(prob[batch_idx])

                for batch_idx in range(bsz):
                    action_keys, infos, messages = prompt_processor.process_batch_response(response_dict[batch_idx], [new_obs[i+batch_idx]] * n)
                    if not best_of_n:
                        selected_idx = find_most_common_action(action_keys, infos)
                    else:
                        selected_idx = np.argmax(prob_dict[batch_idx])
                    idx = new_obs_idxs[i:i+bsz][batch_idx]
                    batch_action_dict[idx] = {"action_key": action_keys[selected_idx], "info": infos[selected_idx]}
                    batch_msg[idx] = new_input_msg[batch_idx]
                    batch_msg[idx].append(messages[selected_idx])

    return batch_action_dict, batch_msg, batch_obs


def batch_interact_environment(agent, env, prompt_processor, num_trajectories, post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x, safe_batch_size = 4, gamma = 0.95, iter=0):
    """
    interact with the environments to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    if iter > 0:
        # Wait for resources to stabilize between batches
        time.sleep(5)
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    bsize = env.batch_size
    all_trajectories = []
    total_score = 0
    total_steps = 0
    total_success_steps = 0
    colorful_print(f"# start collecting {num_trajectories} trajectories", "green")
    
    # import IPython; IPython.embed()
    for num_t in tqdm(range(math.ceil(num_trajectories / bsize))): 
        try:
            done = False
            trajectories = [[] for _ in range(bsize)]
            reset_success = torch.Tensor([False,]).to("cuda:0")
            batch_obs = [None for _ in range(bsize)]
            batch_msg = [[] for _ in range(bsize)]
            batch_past_actions = ['' for _ in range(bsize)]
            batch_eval_info = [{} for _ in range(bsize)]
            batch_done = torch.Tensor([False,]*bsize).to("cuda:0")

            plan_prompts = []
            if deepspeed.comm.get_rank() == 0:
                results = env.reset(num_t)
                batch_obs = [r[0] if r is not None else None for r in results]
                reset_success[0] = True
                    
            steps = 0
            while not all(batch_done):
                steps += 1
                if deepspeed.comm.get_rank() == 0:
                    start_time = time.time()
                    colorful_print(f"## environment steps {str(steps)} of max step {env.max_iter}, is test {env.is_test}", "green")
                
                # Get actions from the agent
                
                batch_action, batch_msg, batch_obs = safe_batch_get_action(agent, prompt_processor, batch_obs, batch_done, batch_msg, batch_past_actions, steps, safe_batch_size = safe_batch_size)

                assert len(batch_action) == bsize
                
                start_step_time = time.time()
                colorful_print(f"## time taken to get action: {start_step_time - start_time}", "green")
                batch_return = env.step(batch_action)
                colorful_print(f"## time taken to step the environment: {time.time() - start_step_time}", "green")
                
                for i, result in zip(range(bsize), batch_return):
                    
                    if result is None:
                        batch_done[i] = True
                        continue
                    next_obs, r, done, info = result
                    cur_action = deepcopy(batch_msg[i][-1]["content"][0]["text"])
                    
                    # Create observation with the current message state (before action)
                    current_obs = deepcopy(batch_obs[i])
                    if batch_msg[i] is not None:
                        current_obs['message'] = deepcopy(batch_msg[i][:-1])
                    
                    # Create next observation with the updated message state from environment step
                    next_obs_copy = deepcopy(next_obs)
                    
                    trajectories[i].append(
                        {
                            "observation": current_obs, 
                            "next_observation": next_obs_copy,
                            "reward": r, 
                            "done": done, 
                            "action": cur_action, 
                            "info":  info,
                            "answer": info['answer'] if info and 'answer' in info else None,
                            "reference_answer": info['reference_answer'] if info and 'reference_answer' in info else None        
                        }
                    )
                    if env.verbose:
                        print(batch_action[i],info)
                    if batch_action[i]["action_key"] is not None and "\nAction:" in cur_action and info["action_success"]:
                        batch_past_actions[i] += cur_action + "\n\n"
                    batch_obs[i] = next_obs
                    batch_done[i] = done
                    if done: 
                        batch_eval_info[i] = info
                    
            colorful_print(f"## start evaluation", "green")
            
            if env.use_webarena_eval:
                trajectories = webarena_batch_eval(trajectories, batch_obs, batch_eval_info, env)              
                                                        
            else:
                # Create a list to hold evaluation message requests and track their original indices
                batch_eval_msg = []
                eval_indices = []
                
                # Only evaluate trajectories where ANSWER was given
                for i in range(len(batch_eval_info)):
                    if (len(trajectories[i]) > 0 and batch_obs[i] is not None and 
                        batch_eval_info[i] is not None and 'answer' in batch_eval_info[i]):
                        try:
                            msg = prompt_processor.process_evaluation(batch_obs[i], batch_eval_info[i])
                            batch_eval_msg.append(msg)
                            eval_indices.append(i)
                        except Exception as e:
                            if env.verbose:
                                logging.error(f"Error processing evaluation for observation {i}: {str(e)}")
                
                # Only call the model if we have valid evaluation requests
                if batch_eval_msg:
                    batch_eval_response, _ = agent.get_action(batch_eval_msg)
                    
                    # Map responses back to original trajectories using eval_indices
                    for msg_idx, traj_idx in enumerate(eval_indices):
                        if msg_idx < len(batch_eval_response):
                            eval_response = batch_eval_response[msg_idx]
                            if env.verbose:
                                logging.info(f"[EVAL RESPONSE for trajectory {traj_idx}] {eval_response}")
                            
                            trajectories[traj_idx][-1]['eval_info'] = eval_response
                            auto_eval_res = 1 if ("SUCCESS" in eval_response and "NOT SUCCESS" not in eval_response) else 0
                            trajectories[traj_idx][-1]['reward'] = auto_eval_res
                
            for i in range(len(trajectories)):
                if len(trajectories[i]) > 0 and 'reward' in trajectories[i][-1].keys():
                    total_score += trajectories[i][-1]['reward']
                    total_steps += len(trajectories[i])

                    # Process trajectory - break this down to add max_steps
                    traj = add_mc_return(add_trajectory_reward(trajectories[i]), gamma=gamma)

                    # Add max_steps to each step in the trajectory
                    for d in traj:
                        d.update({"max_steps": env.max_iter})

                    all_trajectories.append(post_f(traj))
                    
                    if trajectories[i][-1]['reward']:
                        total_success_steps += len(trajectories[i])
                        task = trajectories[i][-1]['observation']['task']
                        all_actions = batch_past_actions[i]
    
            total_try = len(all_trajectories)
            success_rate = total_score / len(all_trajectories)
            avg_step = total_steps / len(all_trajectories)
            if total_score:
                avg_success_step = total_success_steps / total_score
            else:
                avg_success_step = "N/A"

            colorful_print(f"## results: is test {env.is_test} total success {total_score}, total try {total_try}, success rate {success_rate}, avg steps per traj {avg_step}, avg steps per pos traj {avg_success_step}", "green")
            
            
        except Exception as e:
            logging.error(f"[ERROR] DATA COLLECTION")
            import traceback
            logging.error(traceback.format_exc())
            continue

    for env_ in env.envs:
        if env_.driver_task is not None:
            try:
                env_.driver_task.quit()
                dt = env_.driver_task
                env_.driver_task = None
                del dt
                
            except:
                pass

    return all_trajectories
