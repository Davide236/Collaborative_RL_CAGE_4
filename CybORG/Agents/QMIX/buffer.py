import numpy as np
import torch
import random
import gc

class ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size, episode_length, alpha=0.6, beta_start=0.4, beta_frames=1000):
        # Initialize buffer with given parameters
        self.capacity = int(capacity)  # Maximum capacity of the buffer
        self.entries = 0  # Tracks the current number of entries in the buffer

        self.batch_size = batch_size  # Size of the batch for training
        self.episode_length = episode_length  # Length of each episode
        self.obs_dims = obs_dims  # List of observation dimensions for each agent
        self.max_obs_dim = np.max(obs_dims)  # Max dimension for observations across agents
        self.n_agents = len(obs_dims)  # Number of agents in the environment
        self.memory_counter = 0  # Counter to track where to store data
        self.full_memory = [None] * self.capacity  # Storage for the replay buffer

        # Priorities for each memory entry (used for prioritized experience replay)
        self.priorities = np.full((self.capacity,), 1e-6, dtype=np.float32)  
        self.alpha = alpha  # Exponent for prioritizing experiences
        self.beta_start = beta_start  # Starting value for importance-sampling
        self.beta_frames = beta_frames  # Number of frames for gradually increasing beta
        self.frame = 1  # Frame counter
        self.beta = beta_start  # Current value of beta

        self.init_episodic_memory(obs_dims)  # Initialize memory for episodic storage
    
    def init_episodic_memory(self, obs_dims):
        # Initialize episodic memory (for storing one episode at a time)
        ep_length = self.episode_length
        self.episodic_obs = []  # Stores observations of the current episode
        self.episodic_new_obs = []  # Stores next observations of the current episode
        for ii in range(self.n_agents):
            self.episodic_obs.append(torch.Tensor(ep_length, obs_dims[ii]))  # Initialize each agent's observations
            self.episodic_new_obs.append(torch.Tensor(ep_length, obs_dims[ii]))  # Initialize next observations for each agent
        self.episodic_acts = torch.Tensor(self.n_agents, ep_length)  # Actions for the current episode
        self.episodic_rewards = torch.Tensor(self.n_agents, ep_length)  # Rewards for the current episode
        self.episodic_dones = torch.Tensor(self.n_agents, ep_length)  # Done flags for each step in the episode
    
    def delete_episodic_memory(self):
        # Clear episodic memory to free up space
        del self.episodic_obs
        del self.episodic_new_obs
        del self.episodic_acts
        del self.episodic_dones
        del self.episodic_rewards
        gc.collect()  # Run garbage collection to free unused memory
        
    def store_episodic(self, obs, acts, rwds, nobs, dones, step):
        # Store data from the current step of the episode
        for ii in range(self.n_agents):
            self.episodic_obs[ii][step] = torch.Tensor(obs[ii])  # Store observation
            self.episodic_new_obs[ii][step] = torch.Tensor(nobs[ii])  # Store next observation
        self.episodic_acts[:, step] = torch.Tensor(acts)  # Store actions
        self.episodic_rewards[:, step] = torch.Tensor(rwds)  # Store rewards
        self.episodic_dones[:, step] = torch.Tensor(dones)  # Store done flags
    
    def append_episodic(self):
        # After episode ends, store the entire episode in the replay buffer
        obs = torch.stack(self.episodic_obs, dim=0)  # Stack agent observations
        obs_next = torch.stack(self.episodic_new_obs, dim=0)  # Stack next agent observations
        actions = torch.Tensor(self.episodic_acts)  # Convert actions to tensor
        reward = torch.Tensor(self.episodic_rewards)  # Convert rewards to tensor
        data = {  # Create data dictionary
            'obs': obs,
            'obs_next': obs_next,
            'rewards': reward,
            'actions': actions,
            'dones': self.episodic_dones,
            'n_step': self.episode_length
        }
        counter = self.memory_counter % self.capacity  # Get the index in the buffer (circular)
        self.full_memory[int(counter)] = data  # Store the episode in the buffer
        self.memory_counter += 1  # Increment memory counter
        max_priority = self.priorities.max() if self.memory_counter > 0 else 1.0  # Set max priority for new data
        self.priorities[counter] = max_priority  # Assign priority to the new entry
        self.delete_episodic_memory()  # Clear episodic memory
        self.init_episodic_memory(self.obs_dims)  # Reinitialize episodic memory for the next episode
    
    def get_probabilities(self):
        # Calculate probabilities for sampling based on priorities
        priorities = self.priorities[:self.memory_counter]  # Get the priorities up to current memory counter
        priorities = np.nan_to_num(priorities, nan=1e-6)  # Ensure no NaN values
        scaled_priorities = priorities ** self.alpha  # Apply alpha exponent to priorities
        sum_priorities = scaled_priorities.sum()  # Sum of scaled priorities

        # Avoid division by zero
        if sum_priorities == 0:
            sum_priorities = 1.0
        
        # Normalize the priorities
        scaled_priorities /= sum_priorities

        return scaled_priorities  # Return normalized probabilities
    
    def get_importance(self, probabilities):
        # Calculate importance sampling weights (used for prioritized experience replay)
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)  # Update beta
        self.frame += 1  # Increment frame counter
        importance = np.power(1.0 / (self.memory_counter * probabilities), self.beta)  # Compute importance
        importance /= importance.max()  # Normalize importance
        return importance  # Return normalized importance
    
    def sample(self, sample_size):
        # Sample a batch of experiences from the buffer
        if not self.ready():  # Check if the buffer has enough data
            return None
        max_mem = min(self.memory_counter, self.capacity)  # Get the actual memory size (could be less than capacity)
        sample_probs = self.get_probabilities()  # Get the probabilities of each experience being sampled
        sample_probs = sample_probs[:max_mem]  # Consider only the valid entries in the buffer

        # Ensure probabilities sum to 1
        if not np.isclose(sample_probs.sum(), 1.0):
            sample_probs /= sample_probs.sum()
        
        sampled_indices = np.random.choice(max_mem, sample_size, replace=False, p=sample_probs)  # Randomly sample
        importance = self.get_importance(sample_probs[sampled_indices])  # Calculate importance weights
        samples = [self.full_memory[idx] for idx in sampled_indices]  # Get the sampled experiences
        return samples, sampled_indices, importance  # Return the samples, indices, and importance
    
    def set_priorities(self, idx, errors, offset=0.1):
        # Update the priorities of the sampled experiences based on their TD errors
        for i, error in zip(idx, errors):
            self.priorities[i] = np.abs(error) + offset  # Set priority based on the error
    
    def ready(self):
        # Check if the buffer is ready (enough data for a batch)
        return self.batch_size <= len([item for item in self.full_memory if item is not None])
