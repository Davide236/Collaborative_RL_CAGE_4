import torch
import torch.nn as nn
import os
from torch.distributions import Categorical

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, lr, eps, fc):
        super(CriticNetwork, self).__init__()
        # Width of the network
        self.fc = fc
        self.state_dim = state_dim
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
    
    # Load the last saved networks
    def load_last_epoch(self, checkpoint):
        print('Loading Last saved Networks......')
        self.critic.load_state_dict(torch.load(checkpoint['network_state_dict']))
        self.critic_optimizer.load_state_dict(torch.load(checkpoint['optimizer_state_dict']))

    # Load both actor and critic network of the agent
    def load_network(self, checkpoint):
        print('Loading Networks......')
        self.critic.load_state_dict(torch.load(checkpoint['network_state_dict']))
        self.critic_optimizer.load_state_dict(torch.load(checkpoint['optimizer_state_dict']))

