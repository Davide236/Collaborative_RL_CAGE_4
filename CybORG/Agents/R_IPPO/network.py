import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

# Actor Network (Policy Network)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Create a single LSTM layer for processing the state inputs
        self.lstm = nn.LSTM(state_dim, self.hidden_size, batch_first=True)
        # Initialize the parameters of the LSTM layer
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)  # Initialize biases to 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)  # Initialize weights using orthogonal initialization
        
        # Hidden layer after LSTM, which processes the LSTM outputs
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.layer_hidden.weight, np.sqrt(2))  # Use orthogonal initialization
        
        # Hidden layer for policy (action) logits
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))  # Orthogonal initialization
        
        # Final policy layer that produces action logits
        self.policy_layer = nn.Linear(in_features=self.hidden_size, out_features=action_dim)
        nn.init.orthogonal_(self.policy_layer.weight, np.sqrt(0.01))  # Small initialization for output layer
        
        self.recurrent_cell = None  # Placeholder for LSTM cell state
    
    # Initial hidden state for LSTM (to be called at the start of each sequence)
    def get_init_state(self, batch_size):
        self.recurrent_cell = (torch.zeros(batch_size, self.hidden_size).unsqueeze(0),
                            torch.zeros(batch_size, self.hidden_size).unsqueeze(0))  # Initialize hidden and cell states to zeros
        
    def forward(self, state, sequence_length = 1):
        if sequence_length == 1:
            # If processing a single time-step state, apply LSTM and squeeze output to remove unnecessary dimensions
            state, self.recurrent_cell = self.lstm(state.unsqueeze(1), self.recurrent_cell)
            state = state.squeeze(1)  # Remove extra dimension added by unsqueeze
        else:
            # For sequence data, process through LSTM
            state, self.recurrent_cell = self.lstm(state, self.recurrent_cell)
            state_shape = tuple(state.size())
            state = state.reshape(state_shape[0]*state_shape[1], state_shape[2])  # Flatten sequence for processing
        
        # Apply a ReLU activation after the hidden layer
        state = F.relu(self.layer_hidden(state))
        # Further process with another ReLU activation before policy head
        policy_head = F.relu(self.lin_policy(state))
        # Compute action distribution (logits) and return a Categorical distribution
        policy_dist = Categorical(logits=self.policy_layer(policy_head))
        return policy_dist  # Return the categorical distribution representing the action probabilities

# Critic Network (Value Network)
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Create a single LSTM layer for processing the state inputs
        self.lstm = nn.LSTM(state_dim, self.hidden_size, batch_first=True)
        # Initialize the parameters of the LSTM layer
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)  # Initialize biases to 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)  # Initialize weights using orthogonal initialization
        
        # Hidden layer after LSTM, which processes the LSTM outputs
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.layer_hidden.weight, np.sqrt(2))  # Orthogonal initialization for weights
        
        # Value-specific layer after hidden layer
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))  # Orthogonal initialization
        
        # Final value layer that outputs the state value (single scalar)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value_layer.weight, 1)  # Orthogonal initialization for output layer
        self.recurrent_cell = None  # Placeholder for LSTM cell state
    
    # Initial hidden state for LSTM (to be called at the start of each sequence)
    def get_init_state(self, batch_size):
        self.recurrent_cell = (torch.zeros(batch_size, self.hidden_size).unsqueeze(0),
                            torch.zeros(batch_size, self.hidden_size).unsqueeze(0))  # Initialize hidden and cell states to zeros
    
    def forward(self, state, sequence_length = 1):
        if sequence_length == 1:
            # If processing a single time-step state, apply LSTM and squeeze output to remove unnecessary dimensions
            state, self.recurrent_cell = self.lstm(state.unsqueeze(1), self.recurrent_cell)
            state = state.squeeze(1)  # Remove extra dimension added by unsqueeze
        else:
            # For sequence data, process through LSTM
            state, self.recurrent_cell = self.lstm(state, self.recurrent_cell)
            state_shape = tuple(state.size())
            state = state.reshape(state_shape[0]*state_shape[1], state_shape[2])  # Flatten sequence for processing
        
        # Apply ReLU activation after hidden layer
        state = F.relu(self.layer_hidden(state))
        # Process the state through the value network (another hidden layer)
        hidden_value = F.relu(self.lin_value(state))
        # Final value prediction (scalar representing the state value)
        value = self.value_layer(hidden_value).reshape(-1)  # Reshape to a 1D tensor representing the value for each state
        return value  # Return the predicted state value
