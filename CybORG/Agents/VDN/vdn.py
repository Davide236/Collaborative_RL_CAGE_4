import torch.nn.functional as F

import torch
import numpy as np
from vdn_net import QNet
from buffer import ReplayBuffer

class VDN():

    def __init__(self, n_agents, n_actions, actor_dims):
        # TODO: Init Hyperparams method
        self.init_hyperparams()
        self.memory = ReplayBuffer(buffer_limit=100000, n_agents=n_agents, obs_space=18)

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.q_network = QNet(actor_dims, n_actions, recurrent=False)
        self.target_network = QNet(actor_dims, n_actions, recurrent=False)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        
    
    def init_hyperparams(self):
        # TODO: Change this
        #self.episode_length = ep_length
        self.gamma = 0.9
        self.lr = 2.5e-4
        self.grad_norm_clip = 5
        self.chunk_size = 10
        self.update_iter = 10
        self.batch_size = 50
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.training_steps = 0
        #self.decay_steps = total_episodes*0.95 # Training Steps in which it takes to decay
        # TODO: test with this
        self.update_interval = 20
        self.hidden = None
    

    def get_actions(self, state):
        action, self.hidden = self.q_network.sample_action(torch.Tensor(state).unsqueeze(0), self.hidden, self.epsilon)
        action = action[0].data.cpu().numpy().tolist()
        return action
    
    def save_memory(self, state, action, reward, next_state, done ):
        self.memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))

    def train(self):
        _chunk_size = self.chunk_size if self.q_network.recurrent else 1
        for _ in range(self.update_iter):
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

    def init_hidden_state(self):
        self.hidden = self.q_network.init_hidden()
    
    def copy_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self, episode, max_episode):
        self.epsilon = max(self.min_epsilon, self.max_epsilon - (self.max_epsilon - self.min_epsilon) * (episode / (0.6 * max_episode)))
