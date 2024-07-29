import numpy as np
import torch
import random
import gc

class ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size, episode_length, alpha=0.6, beta_start=0.4, beta_frames=1000):
        self.capacity = int(capacity)
        self.entries = 0

        self.batch_size = batch_size
        self.episode_length = episode_length
        self.obs_dims = obs_dims
        self.max_obs_dim = np.max(obs_dims)
        self.n_agents = len(obs_dims)
        self.memory_counter = 0
        self.full_memory = [None] * self.capacity
        
        self.priorities = np.full((self.capacity,), 1e-6, dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.beta = beta_start

        self.init_episodic_memory(obs_dims)
    
    def init_episodic_memory(self, obs_dims):
        ep_length = self.episode_length
        self.episodic_obs = []
        self.episodic_new_obs = []
        for ii in range(self.n_agents):
            self.episodic_obs.append(torch.Tensor(ep_length, obs_dims[ii]))
            self.episodic_new_obs.append(torch.Tensor(ep_length, obs_dims[ii]))
        self.episodic_acts = torch.Tensor(self.n_agents, ep_length)
        self.episodic_rewards = torch.Tensor(self.n_agents, ep_length)
        self.episodic_dones = torch.Tensor(self.n_agents, ep_length)
    
    def delete_episodic_memory(self):
        del self.episodic_obs
        del self.episodic_new_obs
        del self.episodic_acts
        del self.episodic_dones
        del self.episodic_rewards
        gc.collect()
        
    def store_episodic(self, obs, acts, rwds, nobs, dones, step):   
        for ii in range(self.n_agents):
            self.episodic_obs[ii][step] = torch.Tensor(obs[ii])
            self.episodic_new_obs[ii][step] = torch.Tensor(nobs[ii])
        self.episodic_acts[:, step] = torch.Tensor(acts)
        self.episodic_rewards[:, step] = torch.Tensor(rwds)
        self.episodic_dones[:, step] = torch.Tensor(dones)
    
    def append_episodic(self):
        obs = torch.stack(self.episodic_obs, dim=0)
        obs_next = torch.stack(self.episodic_new_obs, dim=0)
        actions = torch.Tensor(self.episodic_acts)
        reward = torch.Tensor(self.episodic_rewards)
        data = {
            'obs': obs,
            'obs_next': obs_next,
            'rewards': reward,
            'actions': actions,
            'dones': self.episodic_dones,
            'n_step': self.episode_length
        }
        counter = self.memory_counter % self.capacity
        self.full_memory[int(counter)] = data
        self.memory_counter += 1
        # Set max priority for new episode
        max_priority = self.priorities.max() if self.memory_counter > 0 else 1.0
        self.priorities[counter] = max_priority
        self.delete_episodic_memory()
        self.init_episodic_memory(self.obs_dims)
    
    def get_probabilities(self):
        priorities = self.priorities[:self.memory_counter]
        priorities = np.nan_to_num(priorities, nan=1e-6)  # Ensure no NaNs
        # Priorities = P^alpha
        scaled_priorities = priorities ** self.alpha
        sum_priorities = scaled_priorities.sum()

        # Avoid division by zero
        if sum_priorities == 0:
            sum_priorities = 1.0
        
        # Normalize
        scaled_priorities /= sum_priorities

        return scaled_priorities
    
    # TODO: Implemented, but not used!
    def get_importance(self, probabilities):
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        importance = np.power(1.0 / (self.memory_counter * probabilities), self.beta)
        importance /= importance.max()
        return importance

    def sample(self, sample_size):
        if not self.ready():
            return None
        max_mem = min(self.memory_counter, self.capacity)
        sample_probs = self.get_probabilities()
        sample_probs = sample_probs[:max_mem]  # Only consider probabilities of the current memory size

        # Ensure probabilities sum to 1
        if not np.isclose(sample_probs.sum(), 1.0):
            sample_probs /= sample_probs.sum()
        
        sampled_indices = np.random.choice(max_mem, sample_size, replace=False, p=sample_probs)
        importance = self.get_importance(sample_probs[sampled_indices])
        samples = [self.full_memory[idx] for idx in sampled_indices]
        return samples,sampled_indices, importance
    
    def set_priorities(self, idx, errors, offset=0.1):
        for i, error in zip(idx, errors):
            self.priorities[i] = np.abs(error) + offset
    
    def ready(self):
        return self.batch_size <= len([item for item in self.full_memory if item is not None])
