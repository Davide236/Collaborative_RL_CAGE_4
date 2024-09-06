import torch.nn.functional as F
import torch
import random
import math
import os
import yaml
import torch.nn as nn
import numpy as np

from CybORG.Agents.Messages.message_handler import MessageHandler
from CybORG.Agents.QMIX.qmix_net import QMixNet, AgentNetwork
#from qmix_net import QMixNet, AgentNetwork

class QMix():

    def __init__(self, n_agents, n_actions, obs_space, state_space, episode_length, total_episodes, messages):
        # TODO: Init Hyperparams method
        self.init_hyperparams(episode_length, total_episodes)
        self.init_check_memory()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_space = obs_space
        self.state_space = state_space
        self.agent_networks = [AgentNetwork(self.obs_space[i], self.n_actions[i], self.fc) for i in range(self.n_agents)]
        self.target_agent_networks = [AgentNetwork(self.obs_space[i], self.n_actions[i], self.fc) for i in range(self.n_agents)]
        # Network for summing up the Q-values of agents
        self.qmix_net_eval = QMixNet(self.n_agents, state_space, self.fc)
        self.qmix_net_target = QMixNet(self.n_agents, state_space, self.fc)
        self.agent_optimizers = [torch.optim.Adam(agent.parameters(), lr=self.lr) for agent in self.agent_networks]
        self.mixing_optimizer = torch.optim.Adam(self.qmix_net_eval.parameters(), lr=self.lr)
        # Message setup
        self.use_messages = messages
        self.message_handler = [MessageHandler(message_type=self.message_type, number=agent_number) for agent_number in range(self.n_agents)] 
        #self.device = torch.device('cpu')
    
    def init_check_memory(self):
        self.loss = []
        self.save_path = f'saved_statistics/qmix/{self.message_type}/data_agent_qmix.csv'

    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        for number, network in enumerate(self.agent_networks):
            checkpoint = os.path.join(f'last_networks/r_qmix/{self.message_type}', f'qmix_{number}')
            checkpoint = torch.load(checkpoint)
            network.load_state_dict(checkpoint['network_state_dict'])
            self.agent_optimizers[number].load_state_dict(checkpoint['optimizer_state_dict'])
            self.target_agent_networks[number].load_state_dict(network.state_dict())
        checkpoint = os.path.join(f'last_networks/r_qmix/{self.message_type}', f'mixer')
        checkpoint = torch.load(checkpoint)
        self.qmix_net_eval.load_state_dict(checkpoint['network_state_dict'])
        self.mixing_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        for number, network in enumerate(self.agent_networks):
            checkpoint = os.path.join(f'saved_networks/r_qmix/{self.message_type}', f'qmix_{number}')
            checkpoint = torch.load(checkpoint)
            network.load_state_dict(checkpoint['network_state_dict'])
            self.target_agent_networks[number].load_state_dict(network.state_dict())
            self.agent_optimizers[number].load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint = os.path.join(f'saved_networks/r_qmix/{self.message_type}', f'mixer')
        checkpoint = torch.load(checkpoint)
        self.qmix_net_eval.load_state_dict(checkpoint['network_state_dict'])
        self.mixing_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    def init_hyperparams(self, ep_length, total_episodes):
        # TODO: Change this
        self.episode_length = ep_length
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.gamma = float(params.get('gamma', 0.99))
        self.lr = float(params.get('lr', 2.5e-4))
        self.grad_norm_clip = float(params.get('grad_norm_clip', 0.5))
        self.start_epsilon =float(params.get('start_epsilon', 1)) 
        self.end_epsilon = float(params.get('end_epsilon', 0.01))
        self.start_temperature =float(params.get('start_temperature', 0.5)) 
        self.end_temperature = float(params.get('end_temperature', 0.01))
        self.fc = int(params.get('fc', 256))
        self.update_interval = int(params.get('update_interval', 10))
        self.message_type = params.get('message_type', 'simple')
        self.exploration = params.get('exploration', 'greedy')
        self.training_steps = 0
        self.decay_steps = total_episodes*0.8 # Training Steps in which it takes to decay
    
    # Exponential annealing
    def epsilon_annealing(self):
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-self.training_steps / self.decay_steps)
        return epsilon
    
    
    def reset_hidden_layer(self):
        for network in self.agent_networks:
            network.rnn_hidden = None
    
    def reset_hidden_target(self):
        for network in self.target_agent_networks:
            network.rnn_hidden = None

    def update_target_networks(self):
        for i in range(self.n_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict())
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    def process_batch(self, batch):
        self.reset_hidden_layer()
        self.reset_hidden_target()
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
        target_qs = target_qs.max(dim=-1)[0]
        q_total_target = self.qmix_net_target(target_qs, central_state_next)
        return q_total_eval, q_total_target, rwrd, term

    # TODO: Check this update function
    def train(self, batch, count):
        total_episodes = len(batch)
        self.training_steps += 1
        q_evals, q_targets, rewards, terminated = [], [], [], []
        pred_diff_arr = []
        for i in range(len(batch)):
            q_total_eval, q_total_target, rwrd, term = self.process_batch(batch[i])
            pred_diff = rwrd[0] + torch.abs(q_total_eval - self.gamma*q_total_target)
            pred_diff = pred_diff.detach().numpy()
            aggregated_td_error = pred_diff.mean().item()
            pred_diff_arr.append(aggregated_td_error)
            terminated.append(term)
            rewards.append(rwrd)
            q_evals.append(q_total_eval)
            q_targets.append(q_total_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        q_evals = q_evals.view(total_episodes, self.episode_length, 1)
        q_targets = q_targets.view(total_episodes, self.episode_length, 1)
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards[0].view(total_episodes,self.episode_length,1)
        dones = torch.stack(terminated, dim=1)
        dones = dones[0].view(total_episodes,self.episode_length,1)
        targets = rewards + self.gamma * q_targets * (1 - dones)
        loss = F.mse_loss(q_evals, targets)
        self.mixing_optimizer.zero_grad()
        for opt in self.agent_optimizers:
            opt.zero_grad()
        loss.backward()
        for agent in self.agent_networks:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        torch.nn.utils.clip_grad_norm_(self.qmix_net_eval.parameters(), self.grad_norm_clip)
        self.mixing_optimizer.step()
        for opt in self.agent_optimizers:
            opt.step()
        if count % self.update_interval:
            self.update_target_networks()
        self.loss.append(loss.item())
        return pred_diff_arr
        
        
    
    # Exponential annealing
    def temperature_annealing(self):
        epsilon = self.end_temperature + (self.start_temperature - self.end_temperature) * math.exp(-self.training_steps / self.decay_steps)
        return epsilon
    

    def eps_greedy(self, q_value, agent_idx):
        epsilon = self.epsilon_annealing()
        random_value = random.random()
        # With probability eps, do a random action
        if agent_idx == 4:
            if random_value < epsilon:
                action = random.randint(0, q_value.shape[0]-1)
            else:
                action = torch.argmax(q_value).item()
        else:
            if random_value < epsilon:
                action = random.randint(0, min(q_value.shape[0], 85) - 1)
            else:
                action = torch.argmax(q_value[:85]).item()
        return action
    

    def bolzman_exploration(self, q_value, agent_idx):
        temperature = self.temperature_annealing()
        soft = nn.Softmax(dim=-1)
        # In this case the Q_Value is based only on the state
        if agent_idx == 4:
            prob =  soft(q_value/temperature)
            prob = prob.detach().numpy()
            prob = prob / prob.sum()
        else:
            mask = np.ones_like(q_value.detach().numpy())
            mask[85:] = -np.inf  # Setting a very negative value
            masked_q_value = q_value + torch.tensor(mask, dtype=torch.float32)
            prob =  soft(masked_q_value/temperature)
            prob = prob.detach().numpy()
        action = np.random.choice(self.n_actions[agent_idx], p=prob)
        return action


    def choose_actions(self, observations):
        actions = []
        messages = []
        for i, agent in enumerate(self.agent_networks):
            obs = observations[i]
            q_value = agent(torch.tensor(obs, dtype=torch.float32))
            if self.exploration == 'greedy':
                action = self.eps_greedy(q_value, i)
            else:
                action = self.bolzman_exploration(q_value, i)
            # Add small value to avoid division by 0
            actions.append(action)
            if self.use_messages:
                message  = self.message_handler[i].prepare_message(obs, action)
                messages.append(message)
        return actions, messages