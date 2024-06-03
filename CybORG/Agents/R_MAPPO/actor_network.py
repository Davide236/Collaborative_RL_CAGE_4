import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden_size = 256
        self.recurrent_layers = 1
        # Create 1 Layer of LSTM
        self.lstm = nn.LSTM(state_dim, self.hidden_size, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        # TODO: This can be of different sizes
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.layer_hidden.weight, np.sqrt(2))
        # Hidden policy layer
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))
        # Final policy layer
        self.policy_layer = nn.Linear(in_features=self.hidden_size, out_features=action_dim)
        nn.init.orthogonal_(self.policy_layer.weight, np.sqrt(0.01))
        self.recurrent_cell = None
    
    # Initial hidden state of size 1   
    def get_init_state(self, batch_size):
        self.recurrent_cell = (torch.zeros(batch_size, self.hidden_size).unsqueeze(0),
                            torch.zeros(batch_size, self.hidden_size).unsqueeze(0))
        
    def forward(self, state, sequence_length = 1):
        if sequence_length == 1:
            state, self.recurrent_cell = self.lstm(state.unsqueeze(1), self.recurrent_cell)
            state = state.squeeze(1)
        else:
            state, self.recurrent_cell = self.lstm(state, self.recurrent_cell)
            state_shape = tuple(state.size())
            state = state.reshape(state_shape[0]*state_shape[1], state_shape[2])
        state = F.relu(self.layer_hidden(state))
        policy_head = F.relu(self.lin_policy(state))
        policy_dist = Categorical(logits=self.policy_layer(policy_head))
        return policy_dist