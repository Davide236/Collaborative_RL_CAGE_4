
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgentNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class QMixNet(nn.Module):

    def __init__(self, n_agents: int, state_shape: int):
        super(QMixNet, self).__init__()
        self.qmix_hidden_dim = 256
        self.n_agents = n_agents
        self.state_shape = state_shape
        #print(f'QMIXnet. Agents: {n_agents}, total central space: {state_shape}')
        self.hyper_w1 = nn.Linear(state_shape, n_agents * self.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(state_shape, self.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(state_shape, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1)
                                      )
    
    def forward(self, q_values, states):
        # The shape of states is (batch_size, max_episode_len, state_shape).
        # The passed q_values are three-dimensional, with a shape of (batch_size, max_episode_len, n_agents).

        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)
        # Add bias to weight calculation
        hidden = F.relu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)
        # Matrix to matrix product
        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
