import torch
import torch.nn as nn
import torch.nn.functional as F

class VDN_Net(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(VDN_Net, self).__init__()
        # Input dim = central_state + num of agents
        print(f'Input dimension of VDN: {input_dim}, total number of actions: {num_actions}')
        # TODO: Check input dimension here
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        # TODO: Remove hidden state
        self.fc_value = nn.Linear(128, 1)
        self.fc_advantage = nn.Linear(128, num_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        q_value = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_value