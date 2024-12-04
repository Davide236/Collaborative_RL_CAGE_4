from torch.nn.utils.rnn import pad_sequence
import torch

class ReplayBuffer:
    def __init__(self):
        # Initialize the rollout memory for storing trajectories
        self.init_rollout_memory()
    
    # Initialize the rollout memory to store various episode-specific data
    def init_rollout_memory(self):
        self.observation_mem = []  # List of states observed
        self.actions_mem = []  # List of actions taken
        self.rewards_mem = []  # List of rewards received
        self.terminal_mem = []  # List of termination flags
        self.logprobs_mem = []  # List of log probabilities of actions
        self.state_val_mem = []  # State values as predicted by the critic
        self.state_val_mem_final = []  # State values, final processed form
        self.value_mem_other = []  # Placeholder for alternative value storage
        self.global_observations_mem = []  # Global observations for multi-agent scenarios

        # Initialize memory for a single episode
        self.init_episodic_memory()
    
    # Initialize episodic memory (to store data for a single episode)
    def init_episodic_memory(self):
        self.episodic_rewards = []  # Rewards for the current episode
        self.episodic_termination = []  # Termination flags for the current episode
        self.episodic_state_val = []  # State values for the current episode
        self.episodic_state = []  # Observations for the current episode
        self.episodic_acts = []  # Actions for the current episode
        self.episodic_logprobs = []  # Log probabilities for the current episode
        self.episodic_global_observations_mem = []  # Global observations for the current episode
    
    # Clear the memory for all episodes
    def clear_rollout_memory(self):
        del self.observation_mem[:]  
        del self.actions_mem[:]  
        del self.rewards_mem[:]  
        del self.terminal_mem[:]  
        del self.logprobs_mem[:]  
        del self.state_val_mem[:]  
        del self.global_observations_mem[:]  
        del self.state_val_mem_final[:] 

    # Clear the memory for the current episode
    def clear_episodic(self):
        del self.episodic_rewards[:]  
        del self.episodic_termination[:] 
        del self.episodic_state_val[:] 
        del self.episodic_state[:]  
        del self.episodic_acts[:]  
        del self.episodic_logprobs[:]  
        del self.episodic_global_observations_mem[:]  
        
    # Append data from the current episode to the main rollout memory
    def append_episodic(self):
        # Append the stored episodic rewards, state values, and terminations into the memory
        self.rewards_mem.append(self.episodic_rewards[:])  # Append episodic rewards
        self.state_val_mem.append(self.episodic_state_val[:])  # Append episodic state values
        self.terminal_mem.append(self.episodic_termination[:])  # Append episodic termination flags
        self.observation_mem.append(torch.cat(self.episodic_state[:]))  # Combine and append episodic states
        self.state_val_mem_final.append(torch.cat(self.episodic_state_val[:]))  # Combine and append final state values
        self.actions_mem.append(torch.cat(self.episodic_acts[:]))  # Combine and append episodic actions
        self.logprobs_mem.append(torch.cat(self.episodic_logprobs[:]))  # Combine and append log probabilities
        self.global_observations_mem.append(torch.cat(self.episodic_global_observations_mem[:]))  # Combine and append global observations
        self.clear_episodic()  # Clear episodic memory after appending
    
    # Pad memory to align data sequences for batch processing
    def pad_memory(self):
        # Pad each list of trajectories to create uniform tensor sizes
        obs = pad_sequence(self.observation_mem, batch_first=True, padding_value=0)
        acts = pad_sequence(self.actions_mem, batch_first=True, padding_value=0)
        logprob = pad_sequence(self.logprobs_mem, batch_first=True, padding_value=0) 
        global_obs = pad_sequence(self.global_observations_mem, batch_first=True, padding_value=0) 
        state_values = pad_sequence(self.state_val_mem_final, batch_first=True, padding_value=0).squeeze()  
        acts = acts.view(-1)  # Flatten actions
        logprob = logprob.view(-1)  # Flatten log probabilities
        return obs, acts, logprob, state_values, global_obs
    
    # Save additional rollout data during an episode
    def save_rollout_data(self, reward, terminated):
        self.episodic_rewards.append(reward)  
        self.episodic_termination.append(terminated)  
    
    # Retrieve a batch of data from memory
    def get_batch(self):
        # Extract data and return all relevant tensors and lists
        obs, acts, logprob, state_values, global_obs = self.pad_memory()
        return obs, acts, logprob, state_values, global_obs, self.rewards_mem, self.state_val_mem, self.terminal_mem
    
    # Save data at the beginning of an episode
    def save_beginning_episode(self, final_state, logprob, action, state_value):
        self.episodic_state.append(final_state) 
        self.episodic_logprobs.append(logprob)
        self.episodic_acts.append(action)  
        self.episodic_state_val.append(state_value)  
    
    # Save data at the end of an episode
    def save_end_episode(self, reward, done, global_obs):
        self.episodic_rewards.append(reward) 
        self.episodic_termination.append(done)  
        self.episodic_global_observations_mem.append(global_obs)  
