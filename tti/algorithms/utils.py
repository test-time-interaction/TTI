import numpy as np
from tti.misc import colorful_print

def rollout_statistics_by_websites(trajectories, env, evaluate = False, iid = False):
    web_names = {}
    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue
        if not trajectory[0]["observation"]["web_name"] in web_names:
            web_names[trajectory[0]["observation"]["web_name"]] = []
        web_names[trajectory[0]["observation"]["web_name"]].append(trajectory)
    info = {}
    for web_name, i_trajectories in web_names.items():
        steps_pos_trajectories = []
        for d in i_trajectories:
            if len(d) > 0 and d[0]["trajectory_reward"] > 0:
                steps_pos_trajectories.append(len(d))
        if evaluate:
            info.update({f"evaluate.{web_name}.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in i_trajectories]),\
                        f"evaluate.{web_name}.avg_steps": np.mean([len(d) for d in i_trajectories]),\
                         f"evaluate.{web_name}.avg_steps_pos_trajectories": np.mean(steps_pos_trajectories),\
                         f"evaluate.{web_name}.num_trajectories": len(i_trajectories),
                        f"evaluate.{web_name}.timeouts": np.mean([len(d) >= env.max_iter for d in i_trajectories])})
        else:
            info.update({f"rollout.{web_name}.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in i_trajectories]),\
                        f"rollout.{web_name}.avg_steps": np.mean([len(d) for d in i_trajectories]),\
                         f"rollout.{web_name}.num_trajectories": len(i_trajectories),
                          f"rollout.{web_name}.avg_steps_pos_trajectories": np.mean(steps_pos_trajectories),\
                        f"rollout.{web_name}.timeouts": np.mean([len(d) >= env.max_iter for d in i_trajectories])})
    if iid:
        new_info = {}
        for key, value in info.items():
            new_info[f"iid_{key}"] = value
        return new_info
    return info

def clean_trajectories(trajectories):
    raw_filtered_trajectories = list(filter(lambda x: len(x) > 0, trajectories))
    
    for trajectory in raw_filtered_trajectories:
        for frame in trajectory:
            if "ac_tree" in frame["observation"]:
                del frame["observation"]["ac_tree"]
            if "ac_tree" in frame["next_observation"]:
                del frame["next_observation"]["ac_tree"]
            if "eval_info" in frame:
                del frame["eval_info"]
            if "reference" in frame:
                del frame["reference"]
            for value in frame["observation"].values():
                assert value is not None
            
    return raw_filtered_trajectories

def filter_trajectories(trajectories, top_p=1):
    raw_filtered_trajectories = list(filter(lambda x: len(x) > 0 and x[0]["trajectory_reward"] > 0, trajectories))
    task2trajectories = {}
    for trajectory in raw_filtered_trajectories:
        if "ANSWER [N/A]" in trajectory[-1]['action']:
            continue
        task = trajectory[0]["observation"]["task"]
        if task not in task2trajectories:
            task2trajectories[task] = []
        task2trajectories[task].append(trajectory)
    # filter the trajectories based on the top_p of the length
    filtered_trajectories = []
    for task, i_trajectories in task2trajectories.items():
        if len(i_trajectories) == 1:
            filtered_trajectories.append(i_trajectories[0])
            continue
        all_traj_lens = [len(traj) for traj in i_trajectories]
        cutoff = np.percentile(all_traj_lens, top_p*100)
        filtered_trajectories += [traj for traj in i_trajectories if len(traj) <= cutoff]

    filtered_trajectories = clean_trajectories(filtered_trajectories)
    colorful_print(f"Trajectories: {len(trajectories)}, Filtered Trajectories: {len(filtered_trajectories)}", fg='green')
    return filtered_trajectories

# calculate the fraction of tasks that the current model have a chance of solving
def calculate_upperbound(trajectories):
    all_tasks = {}
    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue
        if trajectory[0]["observation"]["task"] not in all_tasks:
            all_tasks[trajectory[0]["observation"]["task"]] = 0
        all_tasks[trajectory[0]["observation"]["task"]] += trajectory[0]["trajectory_reward"]
    all_rewards = []
    for reward in all_tasks.values():
        if reward > 0:
            all_rewards.append(1)
        else:
            all_rewards.append(0)
    return np.mean(all_rewards)
