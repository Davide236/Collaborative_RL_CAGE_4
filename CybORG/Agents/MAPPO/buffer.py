from torch.nn.utils.rnn import pad_sequence
import torch

class ReplayBuffer:
    def __init__(self):
        # Initialize the memory for storing observations, actions, and rewards for a rollout
        self.init_rollout_memory()
    
    # Initialize the rollout memory to store various episode-specific data
    def init_rollout_memory(self):
        self.observation_mem = []  # Stores the observations from the environment
        self.actions_mem = []  # Stores the actions taken by the agent
        self.rewards_mem = []  # Stores the rewards received by the agent
        self.terminal_mem = []  # Stores the termination flags for each step
        self.logprobs_mem = []  # Stores the log probabilities of actions taken (for policy-based methods)
        self.state_val_mem = []  # Stores the value of the state as predicted by the value network
        self.global_observations_mem = []  # Stores global observations (e.g., shared states in multi-agent settings)
        self.episodic_rewards = []  # Stores the cumulative rewards for the entire episode
        self.episodic_termination = []  # Stores the termination status for the episode
        self.episodic_state_val = []  # Stores the cumulative state values for the episode

    # Clear the memory of a specific rollout (for example, after training or storing a batch)
    def clear_rollout_memory(self):
        # Clear all stored memory for the current rollout
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        del self.global_observations_mem[:]  # Clear the global observations memory
    
    # Clear the episodic memory at the end of an episode
    def clear_episodic(self):
        # Clear all episodic memory (used for storing rewards, terminations, and state values per episode)
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]
    
    # Save the current episode's data into the rollout memory
    def save_episode(self):
        # Append the stored episodic rewards, state values, and terminations into the memory
        self.rewards_mem.append(self.episodic_rewards[:])
        self.state_val_mem.append(self.episodic_state_val[:])
        self.terminal_mem.append(self.episodic_termination[:]) 
        # Clear episodic memory after saving to avoid duplication
        self.clear_episodic()

    # Save the beginning-of-episode data (state, action, log probability, and state value)
    def save_beginning_episode(self, final_state, logprob, action, state_value):
        # Save the initial state, action, log-probability of action, and state value for the first step
        self.observation_mem.append(final_state)
        self.logprobs_mem.append(logprob)
        self.actions_mem.append(action)
        self.episodic_state_val.append(state_value)

    # Save the end-of-episode data (reward, termination status, and global observation)
    def save_end_episode(self, reward, done, global_obs):
        # Save the reward, termination flag, and global observation for the current step in the episode
        self.episodic_rewards.append(reward)
        self.episodic_termination.append(done)
        self.global_observations_mem.append(global_obs)

    # Get a batch of data for training
    def get_batch(self):
        # Prepare a batch of data for training by converting lists to tensors
        obs = torch.cat(self.observation_mem, dim=0)
        global_obs = torch.cat(self.global_observations_mem, dim=0)
        acts = torch.tensor(self.actions_mem, dtype=torch.float)
        logprob = torch.tensor(self.logprobs_mem, dtype=torch.float).flatten()
        # Return the full batch: observations, global observations, actions, log probabilities, rewards, state values, and terminations
        return obs, global_obs, acts, logprob, self.rewards_mem, self.state_val_mem, self.terminal_mem
