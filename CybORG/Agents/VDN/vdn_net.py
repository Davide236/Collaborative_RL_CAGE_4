import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, observation_space, action_space, fc, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.fc = fc
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, self.fc),
                                                                            nn.ReLU(),
                                                                            nn.Linear(self.fc, self.fc),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.fc, self.fc))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.fc, action_space[agent_i]))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], 1, 0)] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.fc)] * self.num_agents
        max_actions = 242  # Assuming 242 is the total number of actions

        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_val = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

            # Apply action mask for the first four agents to limit actions to 85
            if agent_i < 4:
                q_val = q_val[:, :, :86]  # Only keep the first 86 actions (0 to 85)
                padding = (0, max_actions - 86)  # Pad the rest to match max_actions
                q_values[agent_i] = torch.nn.functional.pad(q_val, padding)
            else:
                q_values[agent_i] = q_val

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon)
        action = torch.empty((out.shape[0], out.shape[1],))

        for agent_i in range(self.num_agents):
            if agent_i < 4:
                # Ensure actions for first 4 agents are within the range 0 to 85
                action[mask, agent_i] = torch.randint(0, 86, action[mask, agent_i].shape).float()
                action[~mask, agent_i] = out[~mask, agent_i, :86].argmax(dim=1).float()
            else:
                # No restriction for other agents
                action[mask, agent_i] = torch.randint(0, out.shape[2], action[mask, agent_i].shape).float()
                action[~mask, agent_i] = out[~mask, agent_i].argmax(dim=1).float()
    
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.fc))
