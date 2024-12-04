import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

# Actor Network (Policy Network)
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Create a single LSTM layer for processing the state inputs
        self.lstm = nn.LSTM(state_dim, self.hidden_size, batch_first=True)
        # Initialize the parameters of the LSTM layer
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0) # Initialize biases to 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)# Initialize weights using orthogonal initialization
        # Hidden layer after LSTM, which processes the LSTM outputs
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.layer_hidden.weight, np.sqrt(2))
        
        # Hidden layer for policy (action) logits
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))
        
        # Final policy layer that produces action logits
        self.policy_layer = nn.Linear(in_features=self.hidden_size, out_features=action_dim)
        nn.init.orthogonal_(self.policy_layer.weight, np.sqrt(0.01))
        
        self.recurrent_cell = None # Placeholder for LSTM cell state
    
    # Initial hidden state for LSTM (to be called at the start of each sequence)
    def get_init_state(self, batch_size):
        self.recurrent_cell = (torch.zeros(batch_size, self.hidden_size).unsqueeze(0),
                            torch.zeros(batch_size, self.hidden_size).unsqueeze(0))
        
    def forward(self, state, sequence_length = 1):
        if sequence_length == 1:
            # If processing a single time-step state, apply LSTM and squeeze output to remove unnecessary dimensions
            state, self.recurrent_cell = self.lstm(state.unsqueeze(1), self.recurrent_cell)
            state = state.squeeze(1)
        else:
            # For sequence data, process through LSTM
            state, self.recurrent_cell = self.lstm(state, self.recurrent_cell)
            state_shape = tuple(state.size())
            state = state.reshape(state_shape[0]*state_shape[1], state_shape[2])
        # Apply a ReLU activation after the hidden layer
        state = F.relu(self.layer_hidden(state))
        # Further process with another ReLU activation before policy head
        policy_head = F.relu(self.lin_policy(state))
        # Compute action distribution (logits) and return a Categorical distribution
        policy_dist = Categorical(logits=self.policy_layer(policy_head))
        return policy_dist # Return the categorical distribution representing the action probabilities