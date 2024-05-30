import os

import torch
from torch import Tensor
import random

from qmix_net import QMixNet, AgentNetwork


class QMix():

    def __init__(self, n_agents, n_actions, obs_space, state_space):
        # TODO: Init Hyperparams method
        self.gamma = 0.99
        self.lr = 2.5e-4
        self.grad_norm_clip = 0.5
        self.epsilon = 0.5
        self.target_update_cycle = 1
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

    def update_target_networks(self):
        for i in range(self.n_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict())
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())
         
    # TODO: Check this update function
    def train(self, batch):
        q_evals, q_targets, rewards, terminated = [], [], [], []
        for i in range(len(batch)):
            state = batch[i]['obs']
            next_state = batch[i]['obs_next']
            # CHECK
            rwrd = batch[i]['rewards']
            term = batch[i]['dones']
            terminated.append(term)
            rewards.append(rwrd)
            ###
            # Get and transpose the actions
            actions = batch[i]['actions'].long()
            transposed_tensor = actions.t()
            episode_actions = transposed_tensor#.unsqueeze(-1)
            # Compute the Q Value given the network and the chosen actions
            agent_qs = [agent(state[j]) for j, agent in enumerate(self.agent_networks)]
            agent_qs = torch.stack(agent_qs, dim=1)
            agent_qs = agent_qs.gather(2, episode_actions.unsqueeze(-1)).squeeze(-1)
            q_total_eval = self.qmix_net_eval(agent_qs, state)
            # Compute target Q_value (and optimal action) given target network
            target_qs = [agent(next_state[j]) for j, agent in enumerate(self.target_agent_networks)]
            target_qs = torch.stack(target_qs, dim=1)
            target_qs = target_qs.max(dim=-1)[0]
            q_total_target = self.qmix_net_target(target_qs, next_state)
            q_evals.append(q_total_eval)
            q_targets.append(q_total_target)
            # Save both Q Values
            # q_evals.append(agent_qs)
            # q_targets.append(target_qs)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        ## CHECK
        q_evals = q_evals.view(5, 25, 1)
        q_targets = q_targets.view(5, 25, 1)
        #
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards[0].view(5,25,1)
        dones = torch.stack(terminated, dim=1)
        dones = dones[0].view(5,25,1)
        #print(rewards)
        #print(f'Dones shape: {dones.shape}')
        #print(f'Shape of Q Eval: {q_evals.shape}')
        # Shape: 25,5,1,1 -> Probably the best shape is 5,25,1
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
        # TODO: Do this only in specific intervals
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
            q_value = agent(torch.tensor(obs, dtype=torch.float32))
            if random.random() < self.epsilon:
                action = random.randint(0, q_value.shape[0]-1)
            else:
                action = torch.argmax(q_value).item()
            actions.append(action)
            agent_name = agents_names[i]
            actions_with_name[agent_name] = action
        return actions_with_name, actions
