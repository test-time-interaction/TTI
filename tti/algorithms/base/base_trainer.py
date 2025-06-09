import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import random

class BaseTrainer():
    def __init__(self, agent,
                    lm_lr: float = 1e-5,
                    batch_size: int = 4,
                    max_grad_norm: float = 1.0,):
        """
        beta: coefficient for the bc loss
        """
        super().__init__()
        self.agent = agent
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
    
    def prepare(self):
        return

    def actor_loss(self, observation, action, **kwargs):
        return {}


    def update(self, trajectories, actor_trajectories, iter):
        return {}

    
    def validate(self, trajectories):
        return {}


    def save(self, path):
        return


    def load(self, path):
        return

