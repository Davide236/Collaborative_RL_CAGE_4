from torch.nn.utils.rnn import pad_sequence
import torch

class ReplayBuffer:
    def __init__(self):
        self.init_rollout_memory()
    
    # Initialize the rollout memory
    # Initialize the rollout memory
    def init_rollout_memory(self):
        # Memory used for collecting total trajectories
        self.observation_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.terminal_mem = [] 
        self.logprobs_mem = []
        self.state_val_mem = [] 
        # Memory used to compute the state value in a different way
        self.state_val_mem_final = []
        self.value_mem_other = []
        self.global_observations_mem = []
        # Initialize episodic memory
        self.init_episodic_memory()
    
    # Initialize episodic memory (just a simple way to divide each episode)
    # This memory is then appended to the 'rollout_memory' so that each episode its divided
    def init_episodic_memory(self):
        self.episodic_rewards = [] 
        self.episodic_termination = []
        self.episodic_state_val = []
        self.episodic_state = []
        self.episodic_acts = []
        self.episodic_logprobs = []
        self.episodic_global_observations_mem = []
    
    # Clear the rollout memory
    def clear_rollout_memory(self):
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        del self.global_observations_mem[:]
        del self.state_val_mem_final[:]
    
    # Clear the episodic memory
    def clear_episodic(self):
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]
        del self.episodic_state[:]
        del self.episodic_acts[:]
        del self.episodic_logprobs[:]
        del self.episodic_global_observations_mem[:]
    
    def append_episodic(self):
        self.rewards_mem.append(self.episodic_rewards[:])
        self.state_val_mem.append(self.episodic_state_val[:])
        self.terminal_mem.append(self.episodic_termination[:])
        self.observation_mem.append(torch.cat(self.episodic_state[:]))
        self.state_val_mem_final.append(torch.cat(self.episodic_state_val[:]))
        self.actions_mem.append(torch.cat(self.episodic_acts[:]))
        self.logprobs_mem.append(torch.cat(self.episodic_logprobs[:]))
        self.global_observations_mem.append(torch.cat(self.episodic_global_observations_mem[:]))
        self.clear_episodic()
    
    def pad_memory(self):
        obs = pad_sequence(self.observation_mem, batch_first=True, padding_value=0)
        acts = pad_sequence(self.actions_mem, batch_first=True, padding_value=0)
        logprob = pad_sequence(self.logprobs_mem, batch_first=True, padding_value=0)
        global_obs = pad_sequence(self.global_observations_mem, batch_first=True, padding_value=0)
        #terminal_list = [torch.tensor([1.0 if value else 0.0 for value in seq]) for seq in self.terminal_mem]
        #terminal = torch.cat(terminal_list, dim=0)
        state_values = pad_sequence(self.state_val_mem_final, batch_first=True, padding_value=0).squeeze()
        acts = acts.view(-1)
        logprob = logprob.view(-1)
        return obs, acts, logprob, state_values, global_obs
    
    # Save the rest of the rollout data
    def save_rollout_data(self, reward, terminated):
        self.episodic_rewards.append(reward)
        self.episodic_termination.append(terminated)

    
    def get_batch(self):
        obs, acts, logprob, state_values, global_obs = self.pad_memory()
        return obs, acts, logprob, state_values,global_obs, self.rewards_mem, self.state_val_mem, self.terminal_mem
    
    def save_beginning_episode(self, final_state, logprob, action, state_value):
        self.episodic_state.append(final_state) 
        self.episodic_logprobs.append(logprob)
        self.episodic_acts.append(action) 
        self.episodic_state_val.append(state_value) 
    
    def save_end_episode(self, reward, done, global_obs):
        self.episodic_rewards.append(reward) # Save reward
        self.episodic_termination.append(done) # Save termination
        self.episodic_global_observations_mem.append(global_obs)