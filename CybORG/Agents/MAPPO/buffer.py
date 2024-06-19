from torch.nn.utils.rnn import pad_sequence
import torch

class ReplayBuffer:
    def __init__(self):
        self.init_rollout_memory()
    
    # Initialize the rollout memory
    def init_rollout_memory(self):
        self.observation_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.terminal_mem = [] 
        self.logprobs_mem = []
        self.state_val_mem = [] 
        self.episodic_rewards = [] #
        self.episodic_termination = []
        self.episodic_state_val = []
        self.global_observations_mem = []
    
    # Clear the rollout memory
    def clear_rollout_memory(self):
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        del self.global_observations_mem[:]
    
    # Clear the episodic memory
    def clear_episodic(self):
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]


    
    def get_batch(self):
        obs = torch.cat(self.observation_mem, dim=0)
        global_obs = torch.cat(self.global_observations_mem, dim=0)
        acts = torch.tensor(self.actions_mem, dtype=torch.float)
        logprob = torch.tensor(self.logprobs_mem, dtype=torch.float).flatten()
        return obs, global_obs, acts, logprob, self.rewards_mem, self.state_val_mem, self.terminal_mem
    
    def save_episode(self):
        self.rewards_mem.append(self.episodic_rewards[:])
        self.state_val_mem.append(self.episodic_state_val[:])
        self.terminal_mem.append(self.episodic_termination[:])
        self.clear_episodic()

    def save_beginning_episode(self, final_state, logprob, action, state_value):
        self.observation_mem.append(final_state) 
        self.logprobs_mem.append(logprob)
        self.actions_mem.append(action) 
        self.episodic_state_val.append(state_value)

    
    def save_end_episode(self, reward, done, global_obs):
        self.episodic_rewards.append(reward) # Save reward
        self.episodic_termination.append(done) # Save termination
        self.global_observations_mem.append(global_obs)