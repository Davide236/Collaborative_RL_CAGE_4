import csv
import torch.nn.functional as F

import torch
from torch import Tensor
import random
import math
import numpy as np
from vdn_net import VDN_Net

class VDN():

    def __init__(self, n_agents, n_actions, state_space, episode_length, total_episodes):
        # TODO: Init Hyperparams method
        self.init_hyperparams(episode_length, total_episodes)
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_space = state_space
        self.shared_network = VDN_Net(state_space + len(self.n_agents), n_actions)
        self.target_network = VDN_Net(state_space + len(self.n_agents), n_actions)
        self.optimizer = torch.optim.Adam(self.shared_network.parameters(), lr=self.lr)
        self.role = {}
        # Make a list of the different roles so that we can append them to the obs space
        role_info = torch.eye(len(n_agents))
        print(f'Roles: {role_info}')
        cnt = 0
        for agent in n_agents:
            self.role[agent] = role_info[cnt]
            cnt += 1
        
    
    def init_hyperparams(self, ep_length, total_episodes):
        # TODO: Change this
        self.episode_length = ep_length
        self.gamma = 0.8
        self.lr = 2.5e-4
        self.grad_norm_clip = 0.5
        self.start_epsilon = 1
        self.end_epsilon = 0.01
        self.training_steps = 0
        self.decay_steps = total_episodes*0.95 # Training Steps in which it takes to decay
        # TODO: test with this
        self.update_interval = 20
    
    # Exponential annealing
    def epsilon_annealing(self):
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-self.training_steps / self.decay_steps)
        return epsilon
    
    def transform_observations(self,obs):
        observations = []
        for i in range(3):
            observations.append(obs[f'agent_{i}'])
        return observations
    
    def combine(self, obs, agent):
        obs = self.transform_observations(obs)
        # TODO: Check this, probably combine the full states togeter
        obs_array = np.array(obs).flatten()
        role_tensor = self.role[agent].cpu()
        # Role tensor. Gives a number to each agent
        obs_tensor_cpu = torch.tensor(obs_array)
        return torch.cat((obs_tensor_cpu, role_tensor)).to(dtype=torch.float32)
    
    def update_target_networks(self):
        self.target_network.load_state_dict(self.shared_network.state_dict())

    def process_batch(self, batch):
        state = batch['obs']
        next_state = batch['obs_next']
        # Get reward and termination for the state
        rwrd = batch['rewards']
        term = batch['dones']
        # Get and transpose the actions
        actions = batch['actions'].long()
        episode_actions = actions.t()
        print(f'Obs shape: {state.shape}, acts shape: {actions.shape}, rewards: {rwrd.shape}')
        agent_qs = [self.shared_network(state[j]) for j, _ in enumerate(self.n_agents)]
        agent_qs = torch.stack(agent_qs, dim=1)
        # Target Q Value evaluated also based on the actions taken
        agent_qs = agent_qs.gather(2, episode_actions.unsqueeze(-1)).squeeze(-1)
        # Compute target Q_value (and optimal action) given target network
        target_qs = [self.target_network(next_state[j]) for j, _ in enumerate(self.n_agents)]
        target_qs = torch.stack(target_qs, dim=1)
        target_qs = target_qs.max(dim=-1)[0]
        return agent_qs, target_qs, rwrd, term

    # TODO: Check this update function
    def train(self, batch, count):
        total_episodes = len(batch)
        self.training_steps += 1
        total_q_vals = torch.zeros(total_episodes)
        expected_q_vals = torch.zeros(total_episodes)
        q_evals, q_targets, rewards, terminated = [], [], [], []
        for i in range(len(batch)):
            q_total_eval, q_total_target, rwrd, term, totq, expq = self.process_batch(batch[i])
            total_q_vals += totq
            expected_q_vals += expq
            terminated.append(term)
            rewards.append(rwrd)
            q_evals.append(q_total_eval)
            q_targets.append(q_total_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        q_evals = q_evals.view(total_episodes, self.episode_length, len(self.n_agents))
        q_targets = q_targets.view(total_episodes, self.episode_length, len(self.n_agents))
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards[0].view(total_episodes,self.episode_length,1)
        dones = torch.stack(terminated, dim=1)
        dones = dones[0].view(total_episodes,self.episode_length,1)
        targets = rewards + self.gamma * q_targets * (1 - dones)
        loss = F.mse_loss(q_evals, targets)
        loss1 = F.mse_loss(total_q_vals, expected_q_vals)
        print(f'Loss: {loss}, loss1 : {loss1}')
        # targets = rewards + self.gamma * q_targets * dones
        # td_error = (q_evals - targets.detach())
        # masked_td_error = dones * td_error
        # loss = (masked_td_error ** 2).sum() / dones.sum()
        self.optimizer.zero_grad()
        #print(f'Rewards: {rewards}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared_network.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        if count % self.update_interval:
            self.update_target_networks()
        return loss
        
        
        
    # Choose an action for each agent with probability eps, otherwise select the one with
    # the highest Q Value
    def choose_actions(self, observations):
        q_values = self.shared_network(torch.tensor(observations, dtype=torch.float32))
        epsilon = self.epsilon_annealing()
        random_value = random.random()
        # With probability eps, do a random action
        if random_value < epsilon:
            return random.randint(0, q_values.shape[0]-1)
        return torch.argmax(q_values).item()