import torch
import torch.nn as nn
import os
from torch.distributions import Categorical

class CriticNetwork(nn.Module):
    def __init__(self, large_state_dim,small_state_dim, n_agents, lr, eps, fc):
        super(CriticNetwork, self).__init__()
        # Width of the network
        self.fc = fc
        self.init_checkpoint()
        self.state_dim = large_state_dim + (small_state_dim)*(n_agents-1)
        # Initialize critc network
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, fc),
            nn.ReLU(),
            nn.Linear(fc,fc),
            nn.ReLU(),
            nn.Linear(fc, 1)
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=eps)

    def get_state_value(self, state):
        state_value = self.critic(state) # Compute the state value
        return state_value # Detach extra elements
    
    def save_last_epoch(self):
        print('Saving Networks.....')
        torch.save(self.critic.state_dict(),self.last_checkpoint_file_critic)
    
    # Load the last saved networks
    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        self.critic.load_state_dict(torch.load(self.last_checkpoint_file_critic))

    # Save both actor and critic networks of the agent
    def save_network(self):
        print('Saving Networks.....')
        torch.save(self.critic.state_dict(),self.checkpoint_file_critic)
    
    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        self.critic.load_state_dict(torch.load(self.checkpoint_file_critic))

    # Initialize checkpoint to save the different agents
    def init_checkpoint(self):
        self.checkpoint_file_critic = os.path.join('saved_networks', f'critic_ppo_central')
        self.last_checkpoint_file_critic = os.path.join('last_networks', f'critic_ppo_central')