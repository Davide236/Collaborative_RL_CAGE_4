import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

# This actor-critic implementation follows the implementation of: https://gitlab.com/ngoodger/ppo_lstm/-/blob/master/recurrent_ppo.ipynb
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden_size = 128
        self.recurrent_layers = 1
        # Create 1 Layer of LSTM
        self.lstm = nn.LSTM(state_dim, self.hidden_size, num_layers=self.recurrent_layers)
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_policy_logits = nn.Linear(self.hidden_size, action_dim)
        self.action_dim = action_dim
        self.hidden_cell = None
    
    # Initial hidden state of size 1   
    def get_init_state(self, batch_size):
        self.hidden_cell = (torch.zeros(self.recurrent_layers, batch_size, self.hidden_size),
                            torch.zeros(self.recurrent_layers, batch_size,self.hidden_size))
        
    def forward(self, state):
        batch_size = state.shape[0] # Always = 1
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size)
        state = state.unsqueeze(0)
        # Set new hidden state cell (send to lstm layer the old hidden cell)
        _, self.hidden_cell = self.lstm(state, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        policy_logits_out = self.layer_policy_logits(hidden_out)
        # Get policy distribution
        policy_dist = Categorical(F.softmax(policy_logits_out, dim=1))
        return policy_dist
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.hidden_size = 128
        self.recurrent_layers = 1
        self.layer_lstm = nn.LSTM(state_dim, self.hidden_size, num_layers=self.recurrent_layers)
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_value = nn.Linear(self.hidden_size, 1)
        self.hidden_cell = None
    
    # Initial hidden state of size 1 
    def get_init_state(self, batch_size):
        self.hidden_cell = (torch.zeros(self.recurrent_layers, batch_size, self.hidden_size),
                            torch.zeros(self.recurrent_layers, batch_size, self.hidden_size))
    
    def forward(self, state):
        batch_size = state.shape[0]
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size)
        state = state.unsqueeze(0)
        # Set new hidden state cell (send to lstm layer the old hidden cell)
        _, self.hidden_cell = self.layer_lstm(state, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        # Output the value of the state
        value_out = self.layer_value(hidden_out)
        return value_out#.detach()