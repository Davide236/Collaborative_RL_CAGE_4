
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, fc):
        super(AgentNetwork, self).__init__()
        self.fc = fc
        self.fc1 = nn.Linear(input_dim, self.fc)
        self.fc2 = nn.Linear(self.fc, self.fc)
        self.fc3 = nn.Linear(self.fc, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class QMixNet(nn.Module):

    def __init__(self, n_agents, state_shape, fc):
        super(QMixNet, self).__init__()
        self.qmix_hidden_dim = fc
        self.n_agents = n_agents
        self.state_shape = state_shape
        #print(f'QMIXnet. Agents: {n_agents}, total central space: {state_shape}')
        self.hyper_w1 = nn.Sequential(nn.Linear(state_shape,self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, n_agents * self.qmix_hidden_dim)
                                      )
        self.hyper_w2 = nn.Sequential(nn.Linear(state_shape,self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1 * self.qmix_hidden_dim)
                                      )

        self.hyper_b1 = nn.Linear(state_shape, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1)
                                      )
    
    def forward(self, q_values, states):
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
        #q_total2 = q_total.view(-1, 1)
        return q_total