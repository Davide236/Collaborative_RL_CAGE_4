import torch as T
import torch.nn as nn
import torch.optim as optim

# Critic Network Class: Used to evaluate the value of a state-action pair
class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions):
        # Initialize the Critic Network class
        super(CriticNetwork, self).__init__()
        
        # Compute the size of the input layer (state dimensions + actions from all agents)
        first_layer_size = input_dims + sum(n_actions)  # Observation and actions of all agents

        # Define the layers of the Critic Network: A 3-layer feedforward neural network
        self.layers = nn.Sequential(*[
            nn.Linear(first_layer_size, fc1_dims),  # First fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(fc1_dims, fc1_dims),  # Second fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(fc1_dims, 1),  # Output layer (value for state-action pair)
        ])
        
        # Optimizer to update the network's parameters using Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, action):
        # Forward pass: concatenate state and action and pass through the network
        return self.layers(T.cat([state, action], dim=1))


# Actor Network Class: Used to select actions based on the current state
class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
                 n_actions):
        # Initialize the Actor Network class
        super(ActorNetwork, self).__init__()

        # Define the layers of the Actor Network: A 3-layer feedforward neural network
        self.layers = nn.Sequential(*[
            nn.Linear(input_dims, fc1_dims),  # First fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(fc1_dims, fc1_dims),  # Second fully connected layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(fc1_dims, n_actions),  # Output layer (number of actions)
        ])
        
        # Optimizer to update the network's parameters using Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs):
        # Forward pass: Pass observations through the network to get actions
        return self.layers(obs)
