import numpy as np
from torch import Tensor
import torch
import random
import gc

class ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size, episode_length): # Todo fix types

        self.capacity = int(capacity)
        self.entries = 0

        self.batch_size = batch_size
        self.episode_length = episode_length
        self.obs_dims = obs_dims
        self.max_obs_dim = np.max(obs_dims)
        self.n_agents = len(obs_dims)
        self.memory_counter = 0
        self.full_memory = [None] * self.capacity
        self.init_episodic_memory(obs_dims)
    
    def init_episodic_memory(self, obs_dims):
        ep_length = self.episode_length
        self.episodic_obs = []
        self.episodic_new_obs = []
        for ii in range(self.n_agents):
            self.episodic_obs.append(Tensor(ep_length, obs_dims[ii]))
            self.episodic_new_obs.append(Tensor(ep_length, obs_dims[ii]))
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
        #acts = np.array(acts)    
        self.episodic_dones[:,step] = Tensor(dones)
    

    def append_episodic(self):
        obs = torch.stack(self.episodic_obs, dim=0)
        obs_next = torch.stack(self.episodic_new_obs, dim=0)
        actions = torch.Tensor(self.episodic_acts)
        # NORMALIZE REWARDS
        # min_tensor = torch.min(self.episodic_rewards)
        # max_tensor = torch.max(self.episodic_rewards)
        # normalized_tensor = (self.episodic_rewards - min_tensor) / (max_tensor - min_tensor)
        # normalized_rewards = normalized_tensor * (1.0 - 0.0) + 0.0
        ## Not Normalized rewards
        reward = torch.Tensor(self.episodic_rewards)
        data = {
            'obs': obs,
            'obs_next': obs_next,
            'rewards': reward, #normalized_rewards,
            'actions': actions,
            'dones': self.episodic_dones,
            'n_step': self.episode_length # Change this
        }
        counter = self.memory_counter % self.capacity
        self.full_memory[int(counter)] = data
        self.memory_counter += 1
        self.delete_episodic_memory()
        self.init_episodic_memory(self.obs_dims)
        

    def sample(self, sample_size):
        if not self.ready(): return None
        non_empty_memory = [item for item in self.full_memory if item is not None]
        sampled_episodes = random.sample(non_empty_memory, sample_size)
        #print(len(sampled_episodes))
        return sampled_episodes
    
    def ready(self):
        non_empty_memory = [item for item in self.full_memory if item is not None]
        return (self.batch_size <= len(non_empty_memory))