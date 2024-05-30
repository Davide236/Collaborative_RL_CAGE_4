import numpy as np
from collections import Counter
from torch import Tensor
import torch
import random
import gc


class ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size: int): # Todo fix types

        self.capacity = int(capacity)
        self.entries = 0

        #self.batch_size = batch_size
        
        self.batch_size = 10
        self.obs_dims = obs_dims
        self.max_obs_dim = np.max(obs_dims)
        self.n_agents = len(obs_dims)
        
        self.full_memory = []
        self.init_episodic_memory(obs_dims)
    
    def init_episodic_memory(self, obs_dims):
        ep_length = 25
        self.episodic_obs = []
        self.episodic_new_obs = []
        for ii in range(self.n_agents):
            # Here 25 since it's max Episode length
            self.episodic_obs.append( Tensor(ep_length, obs_dims[ii]) )
            self.episodic_new_obs.append( Tensor(ep_length, obs_dims[ii]) )
        self.episodic_acts = Tensor(self.n_agents, ep_length)
        self.episodic_rewards = Tensor(self.n_agents, ep_length)
        self.episodic_dones = Tensor(self.n_agents, ep_length)
    
    def delete_episodic_memory(self):
        del self.episodic_obs
        del self.episodic_new_obs
        del self.episodic_acts
        del self.episodic_dones
        del self.episodic_rewards
        # Garbace collector
        gc.collect()
        
        
    def store_episodic(self, obs, acts, rwds, nobs, dones, step):   
        for ii in range(self.n_agents):
            self.episodic_obs[ii][step] = Tensor(obs[ii])
            self.episodic_new_obs[ii][step] = Tensor(nobs[ii])
        self.episodic_acts[:,step] = torch.Tensor(acts)
        self.episodic_rewards[:,step] = Tensor(rwds)
        self.episodic_dones[:,step] = Tensor(dones)
    
    # Here make sure that the normalization works (?)
    def min_max_normalize(tensor, min_val=0.0, max_val=1.0):
        min_tensor = torch.min(tensor)
        max_tensor = torch.max(tensor)
        normalized_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)
        normalized_tensor = normalized_tensor * (max_val - min_val) + min_val
        return normalized_tensor 
       
    def append_episodic(self):
        obs = torch.stack(self.episodic_obs, dim=0)
        obs_next = torch.stack(self.episodic_new_obs, dim=0)
        actions = torch.Tensor(self.episodic_acts)
        # NORMALIZE REWARDS
        min_tensor = torch.min(self.episodic_rewards)
        max_tensor = torch.max(self.episodic_rewards)
        normalized_tensor = (self.episodic_rewards - min_tensor) / (max_tensor - min_tensor)
        normalized_rewards = normalized_tensor * (1.0 - 0.0) + 0.0
        ## Not Normalized rewards
        reward = torch.Tensor(self.episodic_rewards)
        data = {
            'obs': obs,
            'obs_next': obs_next,
            'rewards': normalized_rewards,
            'actions': actions,
            'dones': self.episodic_dones,
            'n_step': 25 # Change this
        }
        # print("DAJE")
        # print(data)
        self.full_memory.append(data)
        # TODO: Maybe clear them
        self.delete_episodic_memory()
        self.init_episodic_memory(self.obs_dims)
        

    def sample(self):
        if not self.ready(): return None
        
        total_sample = 5
        sampled_episodes = random.sample(self.full_memory, total_sample)
        #print(len(sampled_episodes))
        return sampled_episodes
    
    def ready(self):
        return (self.batch_size <= len(self.full_memory))