
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class AgentNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AgentNetwork, self).__init__()
        self.layed_dim = 256
        self.fc1 = nn.Linear(input_dim, self.layed_dim)
        self.fc2 = nn.Linear(self.layed_dim, self.layed_dim)
        self.fc3 = nn.Linear(self.layed_dim, output_dim)

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
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_shape, self.qmix_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout regularization
            nn.Linear(self.qmix_hidden_dim, n_agents * self.qmix_hidden_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_shape, self.qmix_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout regularization
            nn.Linear(self.qmix_hidden_dim, self.qmix_hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_shape, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_shape, self.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.qmix_hidden_dim, 1)
        )

        #self.apply(self.init_weights)

    # def init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.zeros_(m.bias)
    
    def get_value(self, q_values, states):
        episode_num = 1
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
    
    def forward(self, values_n, states):
        states = torch.as_tensor(states, dtype=torch.float32)
        states = states.reshape(-1, self.state_shape)
        agent_qs = values_n.reshape(-1, 1, self.n_agents)
        # First layer
        w_1 = torch.abs(self.hyper_w1(states))
        w_1 = w_1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b_1 = self.hyper_b1(states)
        b_1 = b_1.view(-1, 1, self.qmix_hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs, w_1) + b_1)
        # Second layer
        w_2 = torch.abs(self.hyper_w2(states))
        w_2 = w_2.view(-1, self.qmix_hidden_dim, 1)
        b_2 = self.hyper_b2(states)
        b_2 = b_2.view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_2) + b_2
        # Reshape and return
        q_tot = y.view(-1, 1)
        return q_tot
    
    