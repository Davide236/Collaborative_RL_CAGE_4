import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Critic Network (Value Network)
class CriticNetwork(nn.Module):
    def __init__(self,large_state_dim,small_state_dim, n_agents, hidden_size):
        super(CriticNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.state_dim = large_state_dim + (small_state_dim)*(n_agents-1)
        # Create a single LSTM layer for processing the state inputs
        self.lstm = nn.LSTM(self.state_dim, self.hidden_size, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)# Initialize biases to 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0) # Initialize weights using orthogonal initialization
        # Hidden layer after LSTM, which processes the LSTM outputs
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.layer_hidden.weight, np.sqrt(2))
       
        # Value-specific layer after hidden layer
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))
        
        # Final value layer that outputs the state value (single scalar)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value_layer.weight, 1)
        self.recurrent_cell = None # Placeholder for LSTM cell state
    
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
            state = state.reshape(state_shape[0]*state_shape[1], state_shape[2])
        # Apply ReLU activation after hidden layer
        state = F.relu(self.layer_hidden(state))
        # Process the state through the value network (another hidden layer)
        hidden_value = F.relu(self.lin_value(state))
        # Final value prediction (scalar representing the state value)
        value = self.value_layer(hidden_value).reshape(-1) # Reshape to a 1D tensor representing the value for each state
        return value
    