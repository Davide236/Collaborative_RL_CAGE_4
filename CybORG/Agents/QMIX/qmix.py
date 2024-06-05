import os

import torch
from torch import Tensor
import random
import math

from qmix_net import QMixNet, AgentNetwork

# P.S: Something like this can be done for GPU/CPU

# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")

# TODO: A bit different since each agent has their total number of actions
class QMix():

    def __init__(self, n_agents, n_actions, obs_space, state_space):
        # TODO: Init Hyperparams method
        self.init_hyperparams()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_space = obs_space
        self.state_space = state_space
        self.agent_networks = [AgentNetwork(self.obs_space[i], self.n_actions[i]) for i in range(self.n_agents)]
        self.target_agent_networks = [AgentNetwork(self.obs_space[i], self.n_actions[i]) for i in range(self.n_agents)]
        # Network for summing up the Q-values of agents
        self.qmix_net_eval = QMixNet(self.n_agents, state_space)
        self.qmix_net_target = QMixNet(self.n_agents, state_space)
        self.agent_optimizers = [torch.optim.Adam(agent.parameters(), lr=self.lr) for agent in self.agent_networks]
        self.mixing_optimizer = torch.optim.Adam(self.qmix_net_eval.parameters(), lr=self.lr)
        
        #self.device = torch.device('cpu')
    
    def init_hyperparams(self):
        # TODO: Change this
        self.episode_length = 25
        self.gamma = 0.99
        self.lr = 2.5e-4
        self.grad_norm_clip = 0.5
        # TODO: This epsilon should start from 1 and then be annealed until 0.05
        # Can also decay from 0.9 until 0.1
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        self.training_steps = 0
        self.decay_steps = 10000 # Training Steps in which it takes to decay
        # TODO: test with this
        self.update_interval = 20
    
    # Exponential annealing
    def epsilon_annealing(self):
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-self.training_steps / self.decay_steps)
        return epsilon
    
    def update_target_networks(self):
        for i in range(self.n_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict())
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    def process_batch(self, batch):
        state = batch['obs']
        next_state = batch['obs_next']
        # Concatenate the states together (to get central state)
        permuted_tensor = state.permute(1, 0, 2)
        central_state = permuted_tensor.reshape(self.episode_length, -1)
        permuted_tensor = next_state.permute(1, 0, 2)
        central_state_next = permuted_tensor.reshape(self.episode_length, -1)
        # Get reward and termination for the state
        rwrd = batch['rewards']
        term = batch['dones']
        # Get and transpose the actions
        actions = batch['actions'].long()
        transposed_tensor = actions.t()
        episode_actions = transposed_tensor#.unsqueeze(-1)
        # Compute the Q Value given the network and the chosen actions
        agent_qs = [agent(state[j]) for j, agent in enumerate(self.agent_networks)]
        agent_qs = torch.stack(agent_qs, dim=1)
        # Target Q Value evaluated also based on the actions taken
        agent_qs = agent_qs.gather(2, episode_actions.unsqueeze(-1)).squeeze(-1)
        # Evaluate central state and Q Value
        q_total_eval = self.qmix_net_eval(agent_qs, central_state)
        # Compute target Q_value (and optimal action) given target network
        target_qs = [agent(next_state[j]) for j, agent in enumerate(self.target_agent_networks)]
        target_qs = torch.stack(target_qs, dim=1)
        # TODO: Check this
        # target_qs = target_qs.max(dim=2)[0]
        target_qs = target_qs.max(dim=-1)[0]
        q_total_target = self.qmix_net_target(target_qs, central_state_next)
        return q_total_eval, q_total_target, rwrd, term

    # TODO: Check this update function
    def train(self, batch, count):
        self.training_steps += 1
        q_evals, q_targets, rewards, terminated = [], [], [], []
        for i in range(len(batch)):
            q_total_eval, q_total_target, rwrd, term = self.process_batch(batch[i])
            terminated.append(term)
            rewards.append(rwrd)
            q_evals.append(q_total_eval)
            q_targets.append(q_total_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        ## TODO: Change the 5
        q_evals = q_evals.view(5, self.episode_length, 1)
        q_targets = q_targets.view(5, self.episode_length, 1)
        #
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards[0].view(5,self.episode_length,1)
        dones = torch.stack(terminated, dim=1)
        dones = dones[0].view(5,self.episode_length,1)
        # TODO: Change this
        # targets = rewards + self.gamma * q_targets * (1 - dones)
        # loss = F.smooth_l1_loss(q_eval, targets)
        targets = rewards + self.gamma * q_targets * dones
        td_error = (q_evals - targets.detach())
        masked_td_error = dones * td_error
        loss = (masked_td_error ** 2).sum() / dones.sum()
        self.mixing_optimizer.zero_grad()
        for opt in self.agent_optimizers:
            opt.zero_grad()
        for agent in self.agent_networks:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qmix_net_eval.parameters(), self.grad_norm_clip)
        self.mixing_optimizer.step()
        for opt in self.agent_optimizers:
            opt.step()
        # TODO: Check how many intervals should be used
        if count % self.update_interval:
            self.update_target_networks()
        
        
        
    # Choose an action for each agent with probability eps, otherwise select the one with
    # the highest Q Value
    def choose_actions(self, observations):
        # TODO: Change this
        actions_with_name = {}
        agents_names = ['agent_0', 'agent_1', 'agent_2']
        actions = []
        for i, agent in enumerate(self.agent_networks):
            obs = observations[i]
            # In this case the Q_Value is based only on the state
            q_value = agent(torch.tensor(obs, dtype=torch.float32))
            # TODO: Check this
            epsilon = self.epsilon_annealing()
            # With probability eps, do a random action
            if random.random() < epsilon:
                action = random.randint(0, q_value.shape[0]-1)
            else:
                action = torch.argmax(q_value).item()
            actions.append(action)
            agent_name = agents_names[i]
            actions_with_name[agent_name] = action
        return actions_with_name, actions
