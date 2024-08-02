import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, observation_space, action_space, fc, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.fc = fc
        self.max_actions = max(action_space)
        self.agent_features = nn.ModuleList()
        self.agent_grus = nn.ModuleList() if recurrent else None
        self.agent_qs = nn.ModuleList()

        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i]
            self.agent_features.append(nn.Sequential(nn.Linear(n_obs, self.fc),
                                                     nn.ReLU(),
                                                     nn.Linear(self.fc, self.fc),
                                                     nn.ReLU()))
            if recurrent:
                self.agent_grus.append(nn.GRUCell(self.fc, self.fc))
            self.agent_qs.append(nn.Linear(self.fc, action_space[agent_i]))

    def forward(self, obs, hidden):
        q_values = []
        next_hidden = []

        for agent_i in range(self.num_agents):
            x = self.agent_features[agent_i](obs[:, agent_i, :])
            if self.recurrent:
                x = self.agent_grus[agent_i](x, hidden[:, agent_i, :])
                next_hidden.append(x.unsqueeze(1))
            q_val = self.agent_qs[agent_i](x).unsqueeze(1)

            # Apply action mask for the first four agents to limit actions to 85
            if agent_i < 4:
                q_val = q_val[:, :, :86]  # Only keep the first 86 actions (0 to 85)
                padding = (0, self.max_actions - 86)  # Pad the rest to match max_actions
                q_values.append(F.pad(q_val, padding))
            else:
                q_values.append(q_val)

        if self.recurrent:
            return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)
        else:
            return torch.cat(q_values, dim=1), hidden

    def sample_action(self, obs, hidden, temperature):
        q_values, next_hidden = self.forward(obs, hidden)
        actions = []

        for agent_i in range(self.num_agents):
            q_vals = q_values[:, agent_i, :]

            # Apply Boltzmann Softmax
            softmax_probs = F.softmax(q_vals / temperature, dim=-1)
            
            # Ensure actions for first 4 agents are within the range 0 to 85
            if agent_i < 4:
                softmax_probs = softmax_probs[:, :86]
                action = torch.multinomial(softmax_probs, num_samples=1).squeeze(-1)
            else:
                action = torch.multinomial(softmax_probs, num_samples=1).squeeze(-1)
            
            actions.append(action)

        return torch.stack(actions, dim=1), next_hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.fc))
