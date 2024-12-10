
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class AgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, fc):
        super(AgentNetwork, self).__init__()
        # Define the size of the hidden layers
        self.layed_dim = 256
        self.rnn_hidden = None # Hidden recurrent layer
        self.fc1 = nn.Linear(input_dim, self.layed_dim)
        self.fc2 = nn.Linear(self.layed_dim, self.layed_dim)
        self.rnn = nn.GRUCell(self.layed_dim, self.layed_dim)
        self.fc3 = nn.Linear(self.layed_dim, output_dim)

    def forward(self, x):
        # Forward pass through the network with ReLU activations between layers
        x = F.relu(self.fc1(x))  # First layer with ReLU
        x = F.relu(self.fc2(x))  # Second layer with ReLU
        self.rnn_hidden = self.rnn(x, self.rnn_hidden) # Hidden recurrent layer
        q_values = self.fc3(x) # Output layer without activation (raw Q-values)
        return q_values


class QMixNet(nn.Module):
    def __init__(self, n_agents: int, state_shape: int, fc):
        # Initialize the QMIX network for multi-agent learning
        super(QMixNet, self).__init__()
        
        self.qmix_hidden_dim = 256  # Hidden dimension size for the QMIX network
        self.n_agents = n_agents  # Number of agents
        self.state_shape = state_shape  # State space dimension (size of state vector)

        # Define the first hypernetwork for calculating weight matrix w1
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_shape, self.qmix_hidden_dim),  # First layer
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.2),  # Dropout regularization
            nn.Linear(self.qmix_hidden_dim, n_agents * self.qmix_hidden_dim)  # Output layer for weights
        )

        # Define the second hypernetwork for calculating weight matrix w2
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_shape, self.qmix_hidden_dim),  # First layer
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.2),  # Dropout regularization
            nn.Linear(self.qmix_hidden_dim, self.qmix_hidden_dim)  # Output layer for weights
        )

        # Define the bias terms for the QMIX network
        self.hyper_b1 = nn.Linear(state_shape, self.qmix_hidden_dim)  # Bias for first layer
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_shape, self.qmix_hidden_dim),  # First layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(self.qmix_hidden_dim, 1)  # Output layer for bias term
        )

    def forward(self, values_n, states):
        # Forward pass through the QMIX network
        
        states = torch.as_tensor(states, dtype=torch.float32)  # Convert states to tensor
        states = states.reshape(-1, self.state_shape)  # Reshape state tensor to match input shape
        
        agent_qs = values_n.reshape(-1, 1, self.n_agents)  # Reshape Q-values from agents into appropriate dimensions

        # First layer: compute weight matrix w1 and bias b1
        w_1 = torch.abs(self.hyper_w1(states))  # Get weights for first layer (apply absolute value)
        w_1 = w_1.view(-1, self.n_agents, self.qmix_hidden_dim)  # Reshape weights
        b_1 = self.hyper_b1(states)  # Get bias for first layer
        b_1 = b_1.view(-1, 1, self.qmix_hidden_dim)  # Reshape bias
        
        # Compute hidden layer using batch matrix multiplication (bmm)
        hidden = F.elu(torch.bmm(agent_qs, w_1) + b_1)  # ELU activation after the batch matrix multiplication

        # Second layer: compute weight matrix w2 and bias b2
        w_2 = torch.abs(self.hyper_w2(states))  # Get weights for second layer (apply absolute value)
        w_2 = w_2.view(-1, self.qmix_hidden_dim, 1)  # Reshape weights
        b_2 = self.hyper_b2(states)  # Get bias for second layer
        b_2 = b_2.view(-1, 1, 1)  # Reshape bias
        
        # Compute final output using batch matrix multiplication
        y = torch.bmm(hidden, w_2) + b_2  # Compute final output
        
        # Reshape and return the total Q-value (output of QMIX)
        q_tot = y.view(-1, 1)  # Flatten the output to the expected shape (1-dimensional Q-value)
        return q_tot
    
    