from CybORG.Agents.MADDPG.networks import ActorNetwork, CriticNetwork
import torch as T
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions,total_actions, agent_idx, n_agents, gradient_estimator):
        self.init_hyperparameters(n_actions, agent_idx, n_agents)
        self.agent_name = agent_idx
        self.gradient_estimator = gradient_estimator
        # For each agent Initialize and Actor, Critic and Two target networks
        self.actor = ActorNetwork(self.lr, actor_dims, self.fc1, self.fc2, n_actions)
        self.critic = CriticNetwork(self.lr,critic_dims, self.fc1, self.fc2, n_agents, total_actions)
        self.target_actor = ActorNetwork(self.lr, actor_dims, self.fc1, self.fc2, n_actions)
        self.target_critic = CriticNetwork(self.lr,critic_dims, self.fc1, self.fc2, n_agents, total_actions)
        
        self.update_network_parameters(tau=1)
    
    def init_hyperparameters(self, n_actions, agent_idx, n_agents):
        self.lr = 0.01
        self.fc1 = 256
        self.fc2 = 256
        self.gamma = 0.95
        self.tau = 0.01
        self.max_grad_norm = 5.0
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
        # TODO: Change this
        #normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Add small epsilon to avoid division by zero
        normalized_state = state
        state = T.FloatTensor(normalized_state.reshape(1,-1))
        policy = self.actor.forward(state)
        action = self.gradient_estimator(policy, need_gradients=False)
        # print(f'Choosing agent: {self.agent_idx}')
        # print(action)
        # action2 = self.gradient_estimator(policy, need_gradients=False)
        # print(action2)
        selected_action = T.argmax(action, dim=-1)
        return selected_action.detach().item()

    def target_actions(self, state):
        policy = self.target_actor.forward(state)
        action = self.gradient_estimator(policy, need_gradients=False)
        selected_action = T.argmax(action, dim=-1)
        print(f'Choosing agent: {self.agent_idx}')
        print(selected_action)
        # print(selected_action.detach())
        return selected_action.detach()
      
    def learn(self, memory, agent_list):
        # If not enough memory has been filled
        if not memory.ready():
            return
        
        # Get data from memory
        actor_states, global_states, actions, rewards,\
            actor_new_states, new_global_states, dones = memory.sample_buffer()
        
        # Convert data to tensors
        global_states = T.tensor(np.array(global_states), dtype=T.float)#, device=device)
        rewards = T.tensor(np.array(rewards), dtype=T.float)#, device=device)
        new_global_states = T.tensor(np.array(new_global_states), dtype=T.float)#, device=device)
        dones = T.tensor(np.array(dones))#, device=device)
        # Convert agent-specific data to tensors (since the data is different for each agent)
        actor_states = [T.tensor(actor_states[idx], dtype=T.float)
                        for idx in range(len(agent_list))]
        actor_new_states = [T.tensor(actor_new_states[idx], dtype=T.float)
                            for idx in range(len(agent_list))]
        
        # TODO: Change this completely - Move part of the function on the MADDPG file and another part in the agent file
        print(actor_new_states)
        #actions = [T.tensor(actions[idx], dtype=T.float)
                   #for idx in range(len(agent_list))]
        
        # Calculate target values for critic
        with T.no_grad():
            new_actions = T.cat([agent.target_actions(actor_new_states[idx])
                                 for idx, agent in enumerate(agent_list)],
                                dim=1)
            critic_value_ = self.target_critic.forward(
                                new_global_states, new_actions).squeeze()
            critic_value_[dones[:, self.agent_idx]] = 0.0
            target = rewards[:, self.agent_idx] + self.gamma * critic_value_
        # Calculate critic loss
        old_actions = T.cat([actions[idx] for idx in range(len(agent_list))],
                            dim=1)
        critic_value = self.critic.forward(global_states, old_actions).squeeze()
        critic_loss = F.mse_loss(target, critic_value)
        
        # Update critic parameters
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic.optimizer.step()
        
        # Calculate actor loss
        actions[self.agent_idx] = self.actor.forward(
                actor_states[self.agent_idx])
        actions = T.cat([actions[i] for i in range(len(agent_list))], dim=1)
        actor_loss = -self.critic.forward(global_states, actions).mean()
        # Update actor parameters
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()
        
        # Update parameters of the target networks parameters
        self.update_network_parameters()  