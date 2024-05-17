import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions):
        super(CriticNetwork, self).__init__()
        # Get in input observatios (of all agents) and agents actions
        first_layer_size = input_dims + sum(n_actions)
        # obs = 578, action= 570
        self.layers = nn.Sequential(*[
            nn.Linear(first_layer_size, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 1),
        ])
        #self.double()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
 
    def forward(self, state, action):
        return self.layers(T.cat([state, action], dim=1))
        # x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        # x = F.relu(self.fc2(x))
        # q = self.q(x)
        # return q


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
                 n_actions):
        super(ActorNetwork, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(input_dims, fc1_dims), nn.ReLU(),
            nn.Linear(fc1_dims, fc1_dims), nn.ReLU(),
            nn.Linear(fc1_dims, n_actions),
        ])
        # self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # self.pi = nn.Linear(fc2_dims, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        #self.to(self.device)
    # TODO: Check network Activation function here (should it be Tanh?)
    def forward(self, obs):
        return self.layers(obs)