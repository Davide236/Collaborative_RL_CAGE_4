import torch.nn.functional as F
import torch
import numpy as np
# from vdn_net import QNet
# from buffer import ReplayBuffer
from CybORG.Agents.VDN.vdn_net import QNet
from CybORG.Agents.VDN.buffer import ReplayBuffer
import yaml
import os

class VDN():

    def __init__(self, n_agents, n_actions, actor_dims):
        # TODO: Init Hyperparams method
        self.init_hyperparams()
        self.memory = ReplayBuffer(buffer_limit=100000, n_agents=n_agents, obs_space=actor_dims[0])
        self.init_check_memory()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.epsilon = 1
        self.training_steps = 0
        self.hidden = None
        self.q_network = QNet(actor_dims, n_actions,self.fc, recurrent=False)
        self.target_network = QNet(actor_dims, n_actions,self.fc, recurrent=False)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        
    def init_check_memory(self):
        self.loss = []
        self.save_path = f'saved_statistics/vdn/{self.message_type}/data_agent_vdn.csv'
        self.save_best_path = os.path.join(f'saved_networks/vdn/{self.message_type}', f'vdn_net')
        self.save_last_path = os.path.join(f'last_networks/vdn/{self.message_type}', f'vdn_net')

    def load_network(self):
        print('Loading best network....')
        checkpoint = os.path.join(f'saved_networks/vdn/{self.message_type}', f'vdn_net')
        checkpoint = torch.load(checkpoint)
        self.q_network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def load_last_epoch(self):
        print('Loading network....')
        checkpoint = os.path.join(f'last_networks/vdn/{self.message_type}', f'vdn_net')
        checkpoint = torch.load(checkpoint)
        self.q_network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        self.target_network.load_state_dict(self.q_network.state_dict())

    def init_hyperparams(self):
        # TODO: Change this
        #self.episode_length = ep_length
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.gamma = float(params.get('gamma', 0.99))
        self.lr = float(params.get('lr', 2.5e-4))
        self.grad_norm_clip = float(params.get('grad_norm_clip', 0.5))
        self.start_epsilon =float(params.get('start_epsilon', 1)) 
        self.end_epsilon = float(params.get('end_epsilon', 0.01))
        self.fc = int(params.get('fc', 256))
        self.update_interval = int(params.get('update_interval', 10))
        self.chunk_size = int(params.get('chunk_size', 10))
        self.training_epochs = int(params.get('training_epochs', 10))
        self.batch_size = int(params.get('batch_size', 50))
        self.message_type = params.get('message_type', 'simple')
    

    def get_actions(self, state):
        action, self.hidden = self.q_network.sample_action(torch.Tensor(state).unsqueeze(0), self.hidden, self.epsilon)
        action = action[0].data.cpu().numpy().tolist()
        return action
    
    def save_memory(self, state, action, reward, next_state, done ):
        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

    def train(self):
        _chunk_size = self.chunk_size if self.q_network.recurrent else 1
        for _ in range(self.training_epochs):
            s, a, r, s_prime, done = self.memory.sample_chunk(self.batch_size, _chunk_size)

            hidden = self.q_network.init_hidden(self.batch_size)
            target_hidden = self.target_network.init_hidden(self.batch_size)
            loss = 0
            for step_i in range(_chunk_size):
                q_out, hidden = self.q_network(s[:, step_i, :, :], hidden)
                q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
                sum_q = q_a.sum(dim=1, keepdims=True)

                max_q_prime, target_hidden = self.target_network(s_prime[:, step_i, :, :], target_hidden.detach())
                max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
                target_q = r[:, step_i, :].sum(dim=1, keepdims=True)
                target_q += self.gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i])

                loss += F.smooth_l1_loss(sum_q, target_q.detach())
            
                done_mask = done[:, step_i].squeeze(-1).bool()
                hidden[done_mask] = self.q_network.init_hidden(len(hidden[done_mask]))
                target_hidden[done_mask] = self.target_network.init_hidden(len(target_hidden[done_mask]))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_norm_clip, norm_type=2)
        self.optimizer.step()
        self.loss.append(loss.item())

    def init_hidden_state(self):
        self.hidden = self.q_network.init_hidden()
    
    def copy_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self, episode, max_episode):
        self.epsilon = max(self.end_epsilon, self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (episode / (0.6 * max_episode)))
