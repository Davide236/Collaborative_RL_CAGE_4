import numpy as np
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
        self.episodic_rewards = []  # Stores the cumulative rewards for the entire episode
        self.episodic_termination = []  # Stores the termination status for the episode
        self.episodic_state_val = []  # Stores the cumulative state values for the episode
        self.episodic_intrinsic_rew = []  # Stores any intrinsic rewards for exploration (optional)
        self.intrinsic_rewards_mem = []  # Stores intrinsic rewards for each step

    # Clear the memory of a specific rollout (for example, after training or storing a batch)
    def clear_rollout_memory(self):
        # Clear all stored memory for the current rollout
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        del self.intrinsic_rewards_mem[:]  # Clear the intrinsic rewards memory

    # Clear the episodic memory at the end of an episode
    def clear_episodic(self):
        # Clear all episodic memory (used for storing rewards, terminations, and state values per episode)
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]
        del self.episodic_intrinsic_rew[:]  # Clear the episodic intrinsic rewards

    # Save the current episode's data into the rollout memory
    def save_episode(self):
        # Append the stored episodic rewards, state values, and terminations into the memory
        self.rewards_mem.append(self.episodic_rewards[:])  # Store the rewards for this episode
        self.state_val_mem.append(self.episodic_state_val[:])  # Store the state values for this episode
        self.terminal_mem.append(self.episodic_termination[:])  # Store the termination flags for this episode
        self.intrinsic_rewards_mem.append(self.episodic_intrinsic_rew[:])  # Store the intrinsic rewards for this episode
        # Clear episodic memory after saving to avoid duplication
        self.clear_episodic()

    # Save the end-of-episode data (reward, intrinsic reward, and termination status)
    def save_end_episode(self, reward, intrinsic_rew, done):
        # Save the reward, intrinsic reward, and termination flag for the current step in the episode
        self.episodic_rewards.append(reward)  # Add the current step's reward to episodic rewards
        self.episodic_termination.append(done)  # Add the current step's termination status (True if done)
        self.episodic_intrinsic_rew.append(intrinsic_rew)  # Store the intrinsic reward for exploration

    # Save the beginning-of-episode data (state, action, log probability, and state value)
    def save_beginning_episode(self, final_state, logprob, action, state_value):
        # Save the initial state, action, log-probability of action, and state value for the first step
        self.observation_mem.append(final_state)  # Append the final state of the environment
        self.logprobs_mem.append(logprob)  # Append the log probability of the action taken (useful for policy gradient)
        self.actions_mem.append(action)  # Append the action taken by the agent
        self.episodic_state_val.append(state_value)  # Store the state value for the episode

    # Get a batch of data for training
    def get_batch(self):
        # Prepare a batch of data for training by converting lists to tensors
        obs = torch.cat(self.observation_mem, dim=0)  # Concatenate all observations into a single tensor
        acts = torch.tensor(self.actions_mem, dtype=torch.float)  # Convert the actions list into a tensor
        logprob = torch.tensor(self.logprobs_mem, dtype=torch.float).flatten()  # Flatten and convert logprobs to tensor
        # Return the full batch: observations, actions, log probabilities, rewards, state values, terminations, and intrinsic rewards
        return obs, acts, logprob, self.rewards_mem, self.state_val_mem, self.terminal_mem, self.intrinsic_rewards_mem
