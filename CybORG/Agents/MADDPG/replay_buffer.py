import numpy as np
import torch
import random
import gc

# Class to implement a prioritized replay buffer (PeR) for experience replay in reinforcement learning
class ReplayBuffer:
    def __init__(self, capacity, obs_dims, batch_size, episode_length, alpha=0.6, beta_start=0.4, beta_frames=1000):
        self.capacity = int(capacity)  # Maximum size of the replay buffer
        self.entries = 0  # Number of entries in the buffer
        self.batch_size = batch_size  # Batch size for sampling
        self.episode_length = episode_length  # Length of an episode
        self.obs_dims = obs_dims  # Dimensions of the observations for each agent
        self.max_obs_dim = np.max(obs_dims)  # Max observation dimension
        self.n_agents = len(obs_dims)  # Number of agents
        self.memory_counter = 0  # Counter for the number of experiences stored
        self.full_memory = [None] * self.capacity  # Initialize buffer as an empty list
        
        # Priorities for the experiences, initialized to a small value
        self.priorities = np.full((self.capacity,), 1e-6, dtype=np.float32)
        
        self.alpha = alpha  # Exponent for prioritizing experiences
        self.beta_start = beta_start  # Initial value of beta for importance sampling
        self.beta_frames = beta_frames  # Total number of frames for beta annealing
        self.frame = 1  # Current frame count
        self.beta = beta_start  # Current value of beta

        # Initialize episodic memory
        self.init_episodic_memory(obs_dims)

    # Initialize episodic memory for each agent
    def init_episodic_memory(self, obs_dims):
        ep_length = self.episode_length
        self.episodic_obs = []  # Stores the observations of each agent
        self.episodic_new_obs = []  # Stores the new observations after the action
        for ii in range(self.n_agents):
            self.episodic_obs.append(torch.Tensor(ep_length, obs_dims[ii]))
            self.episodic_new_obs.append(torch.Tensor(ep_length, obs_dims[ii]))
        
        # Centralized observations (combination of all agent observations)
        self.epsodic_central_obs = torch.Tensor(sum(obs_dims), ep_length)
        self.episodic_central_new_obs = torch.Tensor(sum(obs_dims), ep_length)
        
        # Other episodic variables: actions, rewards, and done flags for each agent
        self.episodic_acts = torch.Tensor(self.n_agents, ep_length)
        self.episodic_rewards = torch.Tensor(self.n_agents, ep_length)
        self.episodic_dones = torch.Tensor(self.n_agents, ep_length)
    
    # Delete episodic memory to free up memory resources
    def delete_episodic_memory(self):
        del self.episodic_obs
        del self.episodic_new_obs
        del self.episodic_acts
        del self.epsodic_central_obs
        del self.episodic_central_new_obs
        del self.episodic_dones
        del self.episodic_rewards
        gc.collect()  # Garbage collect to free memory
    
    # Store a new episode of experiences in episodic memory
    def store_episodic(self, obs, acts, rwds, nobs, dones, central_obs, new_central_obs, step):   
        # Store the current and next observations, actions, rewards, and done flags for each agent
        for ii in range(self.n_agents):
            self.episodic_obs[ii][step] = torch.Tensor(obs[ii])
            self.episodic_new_obs[ii][step] = torch.Tensor(nobs[ii])
        
        # Store centralized observations and other episode details
        self.epsodic_central_obs[:, step] = torch.Tensor(central_obs)
        self.episodic_central_new_obs[:, step] = torch.Tensor(new_central_obs)
        self.episodic_acts[:, step] = torch.Tensor(acts)
        self.episodic_rewards[:, step] = torch.Tensor(rwds)
        self.episodic_dones[:, step] = torch.Tensor(dones)
    
    # Append episodic memory to the replay buffer
    def append_episodic(self):
        # Stack all observations and next observations
        obs = torch.stack(self.episodic_obs, dim=0)
        obs_next = torch.stack(self.episodic_new_obs, dim=0)
        
        # Convert other elements (actions, rewards, etc.) into tensors
        actions = torch.Tensor(self.episodic_acts)
        reward = torch.Tensor(self.episodic_rewards)
        
        # Create a dictionary of the episode data
        data = {
            'obs': obs,
            'obs_next': obs_next,
            'central_obs': self.epsodic_central_obs,
            'central_obs_next': self.episodic_central_new_obs,
            'rewards': reward,
            'actions': actions,
            'dones': self.episodic_dones,
            'n_step': self.episode_length
        }
        
        # Calculate the index for storing the data in the buffer
        counter = self.memory_counter % self.capacity
        self.full_memory[int(counter)] = data
        self.memory_counter += 1
        
        # Set maximum priority for the new episode
        max_priority = self.priorities.max() if self.memory_counter > 0 else 1.0
        self.priorities[counter] = max_priority
        
        # Clear episodic memory to free up space for the next episode
        self.delete_episodic_memory()
        self.init_episodic_memory(self.obs_dims)

    # Get the probabilities for sampling based on experience priorities
    def get_probabilities(self):
        priorities = self.priorities[:self.memory_counter]  # Consider only experiences stored so far
        priorities = np.nan_to_num(priorities, nan=1e-6)  # Replace NaNs with a small value
        
        # Apply exponentiation for prioritization
        scaled_priorities = priorities ** self.alpha
        sum_priorities = scaled_priorities.sum()

        # Prevent division by zero
        if sum_priorities == 0:
            sum_priorities = 1.0
        
        # Normalize priorities so they sum to 1
        scaled_priorities /= sum_priorities

        return scaled_priorities
    
    # Calculate the importance of sampled experiences using importance sampling
    def get_importance(self, probabilities):
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        importance = np.power(1.0 / (self.memory_counter * probabilities), self.beta)
        importance /= importance.max()
        return importance

    # Sample a batch of experiences from the buffer
    def sample(self, sample_size):
        if not self.ready():
            return None
        
        # Determine the maximum number of memories to sample from (up to the buffer capacity)
        max_mem = min(self.memory_counter, self.capacity)
        
        # Get probabilities for each experience in the buffer
        sample_probs = self.get_probabilities()
        sample_probs = sample_probs[:max_mem]  # Limit to the current size of the buffer

        # Ensure probabilities sum to 1
        if not np.isclose(sample_probs.sum(), 1.0):
            sample_probs /= sample_probs.sum()
        
        # Sample indices based on probabilities
        sampled_indices = np.random.choice(max_mem, sample_size, replace=False, p=sample_probs)
        
        # Compute importance sampling weights for the sampled experiences
        importance = self.get_importance(sample_probs[sampled_indices])
        
        # Retrieve the sampled experiences
        samples = [self.full_memory[idx] for idx in sampled_indices]
        
        # Combine the samples into a single batch of data
        samples = self.join_elements(samples)
        return samples, sampled_indices, importance
    
    # Combine a list of individual samples into a single batch
    def join_elements(self, data_list):
        joined_data = {
            'obs': [],
            'obs_next': [],
            'central_obs': [],
            'central_obs_next': [],
            'rewards': [],
            'actions': [],
            'dones': []
        }
        
        # Append each element from the sample list to the corresponding batch list
        for data in data_list:
            for key in joined_data:
                joined_data[key].append(data[key])
        
        # Concatenate each list into a single tensor
        concatenated_data = {key: torch.cat(joined_data[key], dim=1) for key in joined_data}
        return concatenated_data

    # Set the priority for a set of experiences based on their error
    def set_priorities(self, idx, errors, offset=0.1):
        for i, error in zip(idx, errors):
            self.priorities[i] = np.abs(error) + offset
    
    # Check if the buffer is ready to sample (i.e., it has enough experiences)
    def ready(self):
        return self.batch_size <= len([item for item in self.full_memory if item is not None])
