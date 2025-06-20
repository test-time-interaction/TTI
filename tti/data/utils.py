from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from copy import deepcopy
import numpy as np

import random
from torch.utils.data import Dataset

class TrajectoryStepDataset(Dataset):
    def __init__(self, trajs):
        """
        Flatten the list of trajectories and store each step as a sample.
        Each sample contains:
          - 'message': a list of dictionaries from the 'observation' field
          - 'action': a string describing the action taken at that step
        After collecting all samples, we shuffle their order.
        """
        self.samples = []
        for traj in trajs:
            for step in traj:
                # Extract only the fields we're interested in
                sample = {
                    "message": step["observation"]["message"],
                    "action": step["action"]
                }
                self.samples.append(sample)

        # Shuffle the dataset in-place
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Custom collate function to handle the 'message' field
def custom_collate_fn(batch):
    """
    Batch is a list of samples where each sample is a dict with keys "message" and "action".
    Because the "message" field is a list of dictionaries (which cannot be easily stacked into a tensor),
    we simply collect them into a list.
    """
    messages = [item["message"] for item in batch]
    actions = [item["action"] for item in batch]
    return {"message": messages, "action": actions}


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.rewards = None
        self.next_observations = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.mc_returns = None
        self.image_features = None
        self.next_image_features = None

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "action": self.actions[rand_indices],
            "image_features": self.image_features[rand_indices],
            "next_image_features": self.next_image_features[rand_indices],
            "reward": self.rewards[rand_indices],
            "next_observation": self.next_observations[rand_indices],
            "done": self.dones[rand_indices],
            "mc_return": self.mc_returns[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation,
        action,
        image_features: np.ndarray,
        next_image_features: np.ndarray,
        reward: np.ndarray,
        next_observation,
        done: np.ndarray,
        mc_return,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(done, bool):
            done = np.array(done)

        if self.observations is None:
            self.observations = np.array(['']*self.max_size, dtype = 'object')
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            self.image_features = np.empty((self.max_size, *image_features.shape), dtype=image_features.dtype)
            self.next_image_features = np.empty((self.max_size, *next_image_features.shape), dtype=next_image_features.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array(['']*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)

        assert reward.shape == ()
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.image_features[self.size % self.max_size] = image_features
        self.next_image_features[self.size % self.max_size] = next_image_features
        self.actions[self.size % self.max_size] = action
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.dones[self.size % self.max_size] = done
        self.mc_returns[self.size % self.max_size] = mc_return

        self.size += 1
        