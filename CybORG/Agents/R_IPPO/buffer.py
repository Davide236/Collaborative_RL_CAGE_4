from torch.nn.utils.rnn import pad_sequence
import torch

class ReplayBuffer:
    def __init__(self):
        # Initialize memory to store trajectory data
        self.init_rollout_memory()  # Call method to initialize rollout memory
    
    def init_rollout_memory(self):
        # Memory used for collecting total trajectories (state-action-reward sequences)
        self.observation_mem = []  # Store the observations during rollout
        self.actions_mem = []      # Store actions taken during rollout
        self.rewards_mem = []      # Store rewards received during rollout
        self.terminal_mem = []     # Store terminal (done) flags
        self.logprobs_mem = []     # Store log probabilities of taken actions
        self.state_val_mem = []    # Store state values
        self.state_val_mem_final = []  # Store final state values for each episode
        # Initialize episodic memory for storing individual episode data
        self.init_episodic_memory()

    # Initialize episodic memory (stores one episode's worth of data)
    # This memory is then appended to the 'rollout_memory' to separate each episode's data
    def init_episodic_memory(self):
        self.episodic_rewards = []      # Store rewards in an episode
        self.episodic_termination = []  # Store terminal flags in an episode
        self.episodic_state_val = []    # Store state values in an episode
        self.episodic_state = []        # Store states in an episode
        self.episodic_acts = []         # Store actions in an episode
        self.episodic_logprobs = []     # Store log probabilities of actions in an episode

    # Clear all rollout memory
    def clear_rollout_memory(self):
        # Delete all stored data from the rollout memory to prepare for a new rollout
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        del self.state_val_mem_final[:]
    
    # Save the current episode's memory to the rollout memory
    def append_episodic(self):
        # Append the current episode's data (observations, actions, rewards, etc.) to the rollout memory
        self.rewards_mem.append(self.episodic_rewards[:])
        self.terminal_mem.append(self.episodic_termination[:])
        self.observation_mem.append(torch.cat(self.episodic_state[:]))
        self.actions_mem.append(torch.cat(self.episodic_acts[:]))
        self.logprobs_mem.append(torch.cat(self.episodic_logprobs[:]))
        self.state_val_mem_final.append(torch.cat(self.episodic_state_val[:]))
        self.state_val_mem.append(self.episodic_state_val[:])
        # Clear the episodic memory to prepare for the next episode
        self.clear_episodic()

    # Clear episodic memory after it has been saved to the rollout memory
    def clear_episodic(self):
        # Delete all data from the episodic memory
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]
        del self.episodic_state[:]
        del self.episodic_acts[:]
        del self.episodic_logprobs[:]

    # Save individual episode data during the rollout process
    def save_rollout_data(self, reward, terminated):
        # Append the reward and termination flag to the episodic memory
        self.episodic_rewards.append(reward)
        self.episodic_termination.append(terminated)
    
    # Pad the memories of different episodes so they have the same length (for batch processing)
    def pad_memory(self):
        # Pad sequences of observations, actions, log probabilities, and state values to ensure equal lengths
        obs = pad_sequence(self.observation_mem, batch_first=True, padding_value=0)
        acts = pad_sequence(self.actions_mem, batch_first=True, padding_value=0)
        logprob = pad_sequence(self.logprobs_mem, batch_first=True, padding_value=0)
        state_values = pad_sequence(self.state_val_mem_final, batch_first=True, padding_value=0).squeeze()
        
        # Reshape actions and log probabilities for further processing
        acts = acts.view(-1)
        logprob = logprob.view(-1)
        
        # Return padded memory data
        return obs, acts, logprob, state_values
    
    # Retrieve a batch of data for training
    def get_batch(self):
        # Call the padding function to get padded data
        obs, acts, logprob, state_values = self.pad_memory()
        # Return the batch of data (observations, actions, log probabilities, state values, rewards, etc.)
        return obs, acts, logprob, state_values, self.rewards_mem, self.state_val_mem, self.terminal_mem
    
    # Save data for the beginning of an episode
    def save_beginning_episode(self, final_state, logprob, action, state_value):
        # Append the starting state, action, and log probability to the episodic memory
        self.episodic_state.append(final_state)
        self.episodic_logprobs.append(logprob)
        self.episodic_acts.append(action)
        self.episodic_state_val.append(state_value)
    
    # Save data for the end of an episode (rewards and termination flags)
    def save_end_episode(self, reward, done):
        # Append the final reward and termination flag to the episodic memory
        self.episodic_rewards.append(reward) 
        self.episodic_termination.append(done) 
