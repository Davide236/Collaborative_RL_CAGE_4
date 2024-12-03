from CybORG.Agents.MADDPG.networks import ActorNetwork, CriticNetwork
from CybORG.Agents.Messages.message_handler import MessageHandler
import torch as T
from copy import deepcopy
import torch.nn.functional as F
import yaml
import os

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, total_actions, agent_idx, n_agents, gradient_estimator, messages):
        """
        Args:
            actor_dims (int): Dimensions of the actor network input.
            critic_dims (int): Dimensions of the critic network input.
            n_actions (int): Number of actions the agent can take.
            total_actions (int): Total number of possible actions in the environment.
            agent_idx (int): Unique identifier for the agent.
            n_agents (int): Total number of agents in the environment.
            gradient_estimator (function): Function used to estimate the gradient for actions.
            messages (bool): Whether the agent should use messages for communication.

        Returns:
            None

        Explanation:
            Initializes the Agent class by setting up actor and critic networks, target networks, hyperparameters, 
            and message handler. Also initializes agent-specific checkpoint and memory.
        """
        self.init_hyperparameters(n_actions, agent_idx, n_agents)
        self.agent_name = agent_idx
        self.gradient_estimator = gradient_estimator
        self.init_check_memory()
        self.init_checkpoint(self.agent_name)

        # Initialize Actor, Critic, and Target Networks for the agent
        self.actor = ActorNetwork(self.lr, actor_dims, self.fc1, self.fc2, n_actions)
        self.critic = CriticNetwork(self.lr, critic_dims, self.fc1, self.fc2, n_agents, total_actions)
        self.target_actor = ActorNetwork(self.lr, actor_dims, self.fc1, self.fc2, n_actions)
        self.target_critic = CriticNetwork(self.lr, critic_dims, self.fc1, self.fc2, n_agents, total_actions)
        
        # Initialize message handler if messages are to be used
        self.use_messages = messages
        self.message_handler = MessageHandler(message_type=self.message_type, number=agent_idx)
        self.update_network_parameters(tau=1)
    
    def init_check_memory(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            Initializes memory arrays for saving loss data and the path to save statistics.
        """
        self.actor_loss = []  # List to store actor loss
        self.critic_loss = []  # List to store critic loss
        self.save_path = f'saved_statistics/maddpg/{self.message_type}/data_agent_{self.agent_name}.csv'

    def load_last_epoch(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            Loads the last saved networks (actor, critic, and target networks) from disk.
        """
        print('Loading Last saved Networks......')
        actor_checkpoint = T.load(self.last_checkpoint_file_actor)
        critic_checkpoint = T.load(self.last_checkpoint_file_critic)
        self.actor.load_state_dict(actor_checkpoint['network_state_dict'])
        self.critic.load_state_dict(critic_checkpoint['network_state_dict'])
        self.actor.optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic.optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        self.target_actor.load_state_dict(actor_checkpoint['network_state_dict'])
        self.target_critic.load_state_dict(critic_checkpoint['network_state_dict'])

    def load_network(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            Loads the agent's networks (actor, critic, and target networks) from the checkpoint files.
        """
        print('Loading Networks......')
        actor_checkpoint = T.load(self.checkpoint_file_actor)
        critic_checkpoint = T.load(self.checkpoint_file_critic)
        self.actor.load_state_dict(actor_checkpoint['network_state_dict'])
        self.critic.load_state_dict(critic_checkpoint['network_state_dict'])
        self.actor.optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic.optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])
        self.target_actor.load_state_dict(actor_checkpoint['network_state_dict'])
        self.target_critic.load_state_dict(critic_checkpoint['network_state_dict'])

    def init_checkpoint(self, number):
        """
        Args:
            number (int): The agent's unique identifier.

        Returns:
            None

        Explanation:
            Initializes the checkpoint file paths for saving the networks. These paths are based on the agent's number.
        """
        self.checkpoint_file_actor = os.path.join(f'saved_networks/maddpg/{self.message_type}', f'actor_maddpg_{number}')
        self.checkpoint_file_critic = os.path.join(f'saved_networks/maddpg/{self.message_type}', f'critic_maddpg_{number}')
        self.last_checkpoint_file_actor = os.path.join(f'last_networks/maddpg/{self.message_type}', f'actor_maddpg_{number}')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/maddpg/{self.message_type}', f'critic_maddpg_{number}')

    def init_hyperparameters(self, n_actions, agent_idx, n_agents):
        """
        Args:
            n_actions (int): Number of actions the agent can take.
            agent_idx (int): Unique identifier for the agent.
            n_agents (int): Total number of agents in the environment.

        Returns:
            None

        Explanation:
            Loads hyperparameters from a YAML configuration file and initializes the agent's learning rate,
            number of neurons in the hidden layers, gamma (discount factor), tau (soft update factor), 
            and other necessary parameters.
        """
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.lr = float(params.get('lr', 0.01))
        self.fc1 = int(params.get('fc1', 256))
        self.fc2 = int(params.get('fc2', 256))
        self.gamma = float(params.get('gamma', 0.99))
        self.tau = float(params.get('tau', 0.01))
        self.policy_regulariser = float(params.get('policy_regulariser', 0.001))
        self.max_grad_norm = float(params.get('max_grad_norm', 0.5))
        self.message_type = params.get('message_type', 'simple')
        self.n_actions = n_actions
        self.agent_idx = agent_idx
        self.n_agents = n_agents

    def update_network_parameters(self, tau=None):
        """
        Args:
            tau (float, optional): Soft update rate for target networks. Defaults to `self.tau`.

        Returns:
            None

        Explanation:
            Updates the parameters of the target actor and critic networks by applying a soft update.
            The parameters of the target networks are slowly pulled towards the parameters of the main networks
            using a factor of `tau`.
        """
        tau = tau or self.tau
        
        # Update target actor network
        source_actor = self.actor
        destination_actor = self.target_actor
        for param, target in zip(source_actor.parameters(), destination_actor.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)
        
        # Update target critic network
        source_critic = self.critic
        destination_critic = self.target_critic
        for param, target in zip(source_critic.parameters(), destination_critic.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    def choose_action(self, state):
        """
        Args:
            state (numpy array or tensor): The current state of the environment.

        Returns:
            (int): The selected action.
            (list): Any message prepared by the message handler.

        Explanation:
            Chooses an action based on the current state using the actor network. The action is determined 
            by selecting the highest probability action from the output of the actor. If messages are enabled, 
            a message is also prepared.
        """
        normalized_state = state
        final_state = T.FloatTensor(normalized_state.reshape(1, -1))  # Reshape state to a tensor
        policy = self.actor.forward(final_state)  # Get the actor's policy output
        action = self.gradient_estimator(policy, need_gradients=False)  # Estimate action from the policy
        selected_action = T.argmax(action, dim=-1)  # Select the action with the highest probability
        
        # Prepare a message if needed
        message = []
        if self.use_messages:
            message = self.message_handler.prepare_message(state, selected_action.detach().item())
        
        return selected_action.detach().item(), message

    def target_actions(self, state):
        """
        Args:
            state (numpy array or tensor): The current state of the environment.

        Returns:
            (int): The selected action from the target network.

        Explanation:
            Chooses the action based on the target actor network for computing the target Q-values.
            This action is used for target Q-value estimation in the critic's learning process.
        """
        policy = self.target_actor.forward(state)
        action = self.gradient_estimator(policy, need_gradients=False)
        selected_action = T.argmax(action, dim=-1)
        return selected_action.detach()

    def learn_critic(self, obs, new_obs, target_actions, sample_actions, rewards, dones):
        """
        Args:
            obs (tensor): The current state (observation) of the agent.
            new_obs (tensor): The next state (observation) after taking an action.
            target_actions (tensor): The target actions to be taken.
            sample_actions (tensor): The actions sampled from the actor network.
            rewards (tensor): The rewards received by the agent after taking actions.
            dones (tensor): A tensor indicating whether the episode has ended (1 if done, 0 if not).

        Returns:
            (float): The critic's loss value.

        Explanation:
            Performs one step of learning for the critic network. It computes the target Q-values 
            based on the reward and the next state, and then calculates the loss using Mean Squared Error 
            between the predicted Q-values and the target Q-values. The loss is used to update the critic's parameters.
        """
        target_actions = T.concat(target_actions, axis=1)
        sampled_actions = T.concat(sample_actions, axis=1)
        Q_next_target = self.target_critic(new_obs, target_actions)
        Q_target = rewards + (1 - dones) * self.gamma * Q_next_target
        Q_eval = self.critic(obs, sampled_actions)
        loss = F.mse_loss(Q_eval, Q_target.detach())  # Compute MSE loss for critic

        # Backpropagation to update critic
        self.critic.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)  # Gradient clipping
        self.critic.optimizer.step()
        self.critic_loss.append(loss.item())
        return loss.item()

    def learn_actor(self, obs, agent_obs, sampled_actions):
        """
        Args:
            obs (tensor): The current state (observation).
            agent_obs (tensor): The observations of all agents.
            sampled_actions (tensor): The actions taken by the current policy.

        Returns:
            (float): The loss value for the actor network.

        Explanation:
            Performs one step of learning for the actor network. It calculates the loss by 
            evaluating the critic network with the predicted actions. The actor network is then updated
            to minimize this loss. The actor loss is combined with a regularizer to prevent large policy 
            updates.
        """
        policy_outputs = self.actor(agent_obs)
        gs_outputs = self.gradient_estimator(policy_outputs)
        _sampled_actions = deepcopy(sampled_actions)
        _sampled_actions[self.agent_idx] = gs_outputs  # Update the action of the current agent
        actions = T.concat(_sampled_actions, axis=1)  # Combine actions from all agents
        loss = -self.critic(obs, actions).mean()  # Calculate the negative of the critic's evaluation as loss
        loss += (policy_outputs ** 2).mean() * self.policy_regulariser  # Add policy regularizer

        # Backpropagation to update actor
        self.actor.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # Gradient clipping
        self.actor.optimizer.step()
        self.actor_loss.append(loss.item())
        return loss.item()
