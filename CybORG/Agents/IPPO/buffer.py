import numpy as np
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
        self.episodic_rewards = [] 
        self.episodic_termination = []
        self.episodic_state_val = []
        #
        self.episodic_intrinsic_rew = []
        self.intrinsic_rewards_mem = []
    

    # Clear the rollout memory
    def clear_rollout_memory(self):
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        #
        del self.intrinsic_rewards_mem[:]

    # Clear the episodic memory
    def clear_episodic(self):
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]
        # 
        del self.episodic_intrinsic_rew[:]
    
    def save_episode(self):
        self.rewards_mem.append(self.episodic_rewards[:])
        self.state_val_mem.append(self.episodic_state_val[:])
        self.terminal_mem.append(self.episodic_termination[:])
        #
        self.intrinsic_rewards_mem.append(self.episodic_intrinsic_rew[:])
        self.clear_episodic()

    def save_end_episode(self, reward, intrinsic_rew, done):
        self.episodic_rewards.append(reward) # Save reward
        self.episodic_termination.append(done) # Save termination
        #
        self.episodic_intrinsic_rew.append(intrinsic_rew)

    def save_beginning_episode(self, final_state, logprob, action, state_value):
        self.observation_mem.append(final_state) 
        self.logprobs_mem.append(logprob)
        self.actions_mem.append(action) 
        self.episodic_state_val.append(state_value) 

    def get_batch(self):
        obs = torch.cat(self.observation_mem, dim=0)
        acts = torch.tensor(self.actions_mem, dtype=torch.float)
        logprob = torch.tensor(self.logprobs_mem, dtype=torch.float).flatten()
        return obs, acts, logprob, self.rewards_mem, self.state_val_mem, self.terminal_mem, self.intrinsic_rewards_mem