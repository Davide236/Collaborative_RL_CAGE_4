from CybORG.Agents.MADDPG.networks import ActorNetwork, CriticNetwork
import torch as T
from copy import deepcopy
import torch.nn.functional as F
import yaml
import os

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions,total_actions, agent_idx, n_agents, gradient_estimator):
        self.init_hyperparameters(n_actions, agent_idx, n_agents)
        self.agent_name = agent_idx
        self.gradient_estimator = gradient_estimator
        self.init_check_memory()
        self.init_checkpoint(self.agent_name)
        # For each agent Initialize and Actor, Critic and Two target networks
        self.actor = ActorNetwork(self.lr, actor_dims, self.fc1, self.fc2, n_actions)
        self.critic = CriticNetwork(self.lr,critic_dims, self.fc1, self.fc2, n_agents, total_actions)
        self.target_actor = ActorNetwork(self.lr, actor_dims, self.fc1, self.fc2, n_actions)
        self.target_critic = CriticNetwork(self.lr,critic_dims, self.fc1, self.fc2, n_agents, total_actions)
        
        self.update_network_parameters(tau=1)
    
    # Initialize arrays to save important information for the training
    def init_check_memory(self):
        self.actor_loss = []
        self.critic_loss = []
        self.save_path = f'saved_statistics/maddpg/{self.message_type}/data_agent_{self.agent_name}.csv'

    
    # Load the last saved networks
    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        self.actor.load_state_dict(T.load(self.last_checkpoint_file_actor['network_state_dict']))
        self.critic.load_state_dict(T.load(self.last_checkpoint_file_critic['network_state_dict']))
        self.actor.optimizer.load_state_dict(T.load(self.last_checkpoint_file_actor['optimizer_state_dict']))
        self.critic.optimizer.load_state_dict(T.load(self.last_checkpoint_file_critic['optimizer_state_dict']))
        self.target_actor.load_state_dict(T.load(self.last_checkpoint_file_actor))
        self.target_critic.load_state_dict(T.load(self.last_checkpoint_file_critic))

    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        self.actor.load_state_dict(T.load(self.checkpoint_file_actor['network_state_dict']))
        self.critic.load_state_dict(T.load(self.checkpoint_file_critic['network_state_dict']))
        self.actor.optimizer.load_state_dict(T.load(self.checkpoint_file_actor['optimizer_state_dict']))
        self.critic.optimizer.load_state_dict(T.load(self.checkpoint_file_critic['optimizer_state_dict']))
        self.target_actor.load_state_dict(T.load(self.checkpoint_file_actor))
        self.target_critic.load_state_dict(T.load(self.checkpoint_file_critic))

    # Initialize checkpoint to save the different agents
    def init_checkpoint(self, number):
        self.checkpoint_file_actor = os.path.join(f'saved_networks/maddpg/{self.message_type}', f'actor_maddpg_{number}')
        self.checkpoint_file_critic = os.path.join(f'saved_networks/maddpg/{self.message_type}', f'critic_maddpg_{number}')
        self.last_checkpoint_file_actor = os.path.join(f'last_networks/maddpg/{self.message_type}', f'actor_maddpg_{number}')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/maddpg/{self.message_type}', f'critic_maddpg_{number}')


    def init_hyperparameters(self, n_actions, agent_idx, n_agents):
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.lr = float(params.get('lr',0.01))
        self.fc1 = int(params.get('fc1', 256))
        self.fc2 = int(params.get('fc2', 256))
        self.gamma = float(params.get('gamma',0.99)) 
        self.tau = float(params.get('tau',0.01))
        self.policy_regulariser = float(params.get('policy_regulariser',0.001))
        self.max_grad_norm = float(params.get('max_grad_norm',0.5))
        self.message_type = params.get('message_type', 'simple')
        self.n_actions = n_actions
        self.agent_idx = agent_idx
        self.n_agents = n_agents
    

    # Update parameters of target networks
    def update_network_parameters(self, tau=None):
        tau = tau or self.tau
        
        source_actor = self.actor
        destination_actor = self.target_actor
        # Extract parameters from actor (and target) - Parameters of each layer
        for param, target in zip(source_actor.parameters(), destination_actor.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)
        
        # Do the same for critic
        source_critic = self.critic
        destination_critic = self.target_critic

        for param, target in zip(source_critic.parameters(), destination_critic.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)
    
    
    def choose_action(self, state, evaluate=False):
        normalized_state = state
        state = T.FloatTensor(normalized_state.reshape(1,-1))
        policy = self.actor.forward(state)
        action = self.gradient_estimator(policy, need_gradients=False)
        if self.agent_idx != 4:
            max_allowed_action = 85
            mask = T.full(action.shape, float('-inf')).to(action.device)
            action = T.where(T.arange(action.shape[-1]).to(action.device) > max_allowed_action, mask, action)
        selected_action = T.argmax(action, dim=-1)
        return selected_action.detach().item()

    def target_actions(self, state):
        policy = self.target_actor.forward(state)
        action = self.gradient_estimator(policy, need_gradients=False)
        if self.agent_idx != 4:
            max_allowed_action = 85
            mask = T.full(action.shape, float('-inf')).to(action.device)
            action = T.where(T.arange(action.shape[-1]).to(action.device) > max_allowed_action, mask, action)
        selected_action = T.argmax(action, dim=-1)
        return selected_action.detach()


    def learn_critic(self, obs, new_obs, target_actions, sample_actions, rewards, dones):
        target_actions = T.concat(target_actions, axis=1)
        sampled_actions = T.concat(sample_actions, axis=1)
        Q_next_target = self.target_critic(new_obs, target_actions)
        Q_target = rewards + (1 - dones) * self.gamma * Q_next_target
        Q_eval = self.critic(obs, sampled_actions)
        loss = F.mse_loss(Q_eval, Q_target.detach())

        self.critic.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic.optimizer.step()
        self.critic_loss.append(loss.item())
        return loss.item()

    def learn_actor(self, obs, agent_obs, sampled_actions):
        policy_outputs = self.actor(agent_obs)
        gs_outputs = self.gradient_estimator(policy_outputs)
        _sampled_actions = deepcopy(sampled_actions)
        _sampled_actions[self.agent_idx] = gs_outputs
        actions = T.concat(_sampled_actions, axis=1)
        loss = - self.critic(obs, actions).mean()
        # Actor Loss = - Critic_Network(state, actions)
        #loss = - self.critic(obs, actions).mean()
        loss += (policy_outputs ** 2).mean() * self.policy_regulariser

        self.actor.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step() 
        self.actor_loss.append(loss.item())
        return loss.item()