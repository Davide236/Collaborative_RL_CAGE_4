import torch.nn.functional as F
import torch
import random
import math
import os
import yaml

from CybORG.Agents.QMIX.qmix_net import QMixNet, AgentNetwork
#from qmix_net import QMixNet, AgentNetwork

class QMix():

    def __init__(self, n_agents, n_actions, obs_space, state_space, episode_length, total_episodes):
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
        
        #self.device = torch.device('cpu')
    
    def init_check_memory(self):
        self.loss = []
        self.save_path = f'saved_statistics/qmix/{self.message_type}/data_agent_qmix.csv'

    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        for number, network in enumerate(self.agent_networks):
            checkpoint = os.path.join(f'last_networks/qmix/{self.message_type}', f'qmix_{number}')
            checkpoint = torch.load(checkpoint)
            network.load_state_dict(torch.load(checkpoint['network_state_dict']))
            self.agent_optimizers[number].load_state_dict(torch.load(checkpoint['optimizer_state_dict']))
            self.target_agent_networks[number].load_state_dict(network.state_dict())
        checkpoint = os.path.join(f'last_networks/qmix/{self.message_type}', f'mixer')
        checkpoint = torch.load(checkpoint)
        self.qmix_net_eval.load_state_dict(torch.load(checkpoint['network_state_dict']))
        self.mixing_optimizer.load_state_dict(torch.load(checkpoint['optimizer_state_dict']))
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        for number, network in enumerate(self.agent_networks):
            checkpoint = os.path.join(f'saved_networks/qmix/{self.message_type}', f'qmix_{number}')
            checkpoint = torch.load(checkpoint)
            network.load_state_dict(checkpoint['network_state_dict'])
            self.target_agent_networks[number].load_state_dict(network.state_dict())
            self.agent_optimizers[number].load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint = os.path.join(f'saved_networks/qmix/{self.message_type}', f'mixer')
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
        self.fc = int(params.get('fc', 256))
        self.update_interval = int(params.get('update_interval', 10))
        self.message_type = params.get('message_type', 'simple')
        self.training_steps = 0
        self.decay_steps = total_episodes*0.95 # Training Steps in which it takes to decay
        # TODO: test with this
    
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
        target_qs = target_qs.max(dim=-1)[0]
        q_total_target = self.qmix_net_target(target_qs, central_state_next)
        return q_total_eval, q_total_target, rwrd, term

    # TODO: Check this update function
    def train(self, batch, count):
        total_episodes = len(batch)
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
        q_evals = q_evals.view(total_episodes, self.episode_length, 1)
        q_targets = q_targets.view(total_episodes, self.episode_length, 1)
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards[0].view(total_episodes,self.episode_length,1)
        dones = torch.stack(terminated, dim=1)
        dones = dones[0].view(total_episodes,self.episode_length,1)
        targets = rewards + self.gamma * q_targets * (1 - dones)
        loss2 = F.smooth_l1_loss(q_evals, targets)
        loss = F.mse_loss(q_evals, targets)
        # targets = rewards + self.gamma * q_targets * dones
        # td_error = (q_evals - targets.detach())
        # masked_td_error = dones * td_error
        # loss = (masked_td_error ** 2).sum() / dones.sum()
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
        
        
        
    # Choose an action for each agent with probability eps, otherwise select the one with
    # the highest Q Value
    def choose_actions(self, observations):
        # TODO: Change this
        actions = []
        for i, agent in enumerate(self.agent_networks):
            obs = observations[i]
            # In this case the Q_Value is based only on the state
            q_value = agent(torch.tensor(obs, dtype=torch.float32))
            # TODO: Check this
            epsilon = self.epsilon_annealing()
            random_value = random.random()
            # With probability eps, do a random action
            if i == 4:
                if random_value < epsilon:
                    action = random.randint(0, q_value.shape[0]-1)
                else:
                    action = torch.argmax(q_value).item()
            else:
                if random_value < epsilon:
                    action = random.randint(0, min(q_value.shape[0], 85) - 1)
                else:
                    action = torch.argmax(q_value[:85]).item()
            actions.append(action)
        return actions