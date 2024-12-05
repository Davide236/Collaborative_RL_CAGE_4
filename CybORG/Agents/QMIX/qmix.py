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

class QMix():
    def __init__(self, n_agents, n_actions, obs_space, state_space, episode_length, total_episodes, messages):
        """
        Args:
            n_agents (int): Number of agents.
            n_actions (list): Actions available to each agent.
            obs_space (list): Observation space for each agent.
            state_space (int): Global state space.
            episode_length (int): Length of an episode.
            total_episodes (int): Total number of episodes for training.
            messages (bool): Flag to use message passing.
        
        Returns:
            None
        
        Explanation:
            Initializes the parameters for the QMIX model, including network setup and optimizers.
        """
        self.init_hyperparams(episode_length, total_episodes)  # Initialize hyperparameters (learning rate, gamma, etc.)
        self.init_check_memory()  # Initialize memory for loss tracking and saving statistics
        self.n_agents = n_agents  # Set the number of agents
        self.n_actions = n_actions  # Set the actions available to each agent
        self.obs_space = obs_space  # Set the observation space for each agent
        self.state_space = state_space  # Set the global state space
        # Initialize the agent networks, one for each agent
        self.agent_networks = [AgentNetwork(self.obs_space[i], self.n_actions[i], self.fc) for i in range(self.n_agents)]
        # Initialize the target agent networks (for target Q-value computation)
        self.target_agent_networks = [AgentNetwork(self.obs_space[i], self.n_actions[i], self.fc) for i in range(self.n_agents)]
        # Initialize the QMIX network for mixing the Q-values of individual agents
        self.qmix_net_eval = QMixNet(self.n_agents, state_space, self.fc)
        # Initialize the target QMIX network for target Q-value calculation
        self.qmix_net_target = QMixNet(self.n_agents, state_space, self.fc)
        # Optimizer for each agent network
        self.agent_optimizers = [torch.optim.Adam(agent.parameters(), lr=self.lr) for agent in self.agent_networks]
        # Optimizer for the QMIX network
        self.mixing_optimizer = torch.optim.Adam(self.qmix_net_eval.parameters(), lr=self.lr)
        self.use_messages = messages  # Flag to indicate if messages are passed between agents
        # Initialize message handlers for each agent
        self.message_handler = [MessageHandler(message_type=self.message_type, number=agent_number) for agent_number in range(self.n_agents)] 

    def init_check_memory(self):
        """
        Args:
            None
            
        Returns:
            None
        
        Explanation:
            Initializes the memory for loss tracking and saving statistics.
        """
        self.loss = []  # List to track the loss values over training
        # Path to save the statistics and data related to training
        self.save_path = f'saved_statistics/qmix/{self.message_type}/data_agent_qmix.csv'

    def load_last_epoch(self):
        """
        Args:
            None
            
        Returns:
            None
        
        Explanation:
            Loads the weights and optimizer states for both agent and QMIX networks from the most recent checkpoint 
            for continued training.
        """
        print('Loading Last saved Networks......')
        for number, network in enumerate(self.agent_networks):
            # Load the checkpoint for each agent network
            checkpoint = os.path.join(f'last_networks/qmix/{self.message_type}', f'qmix_{number}')
            checkpoint = torch.load(checkpoint)  # Load the checkpoint from the file
            network.load_state_dict(checkpoint['network_state_dict'])  # Load the network state
            self.agent_optimizers[number].load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
            self.target_agent_networks[number].load_state_dict(network.state_dict())  # Copy current network weights to target network
        # Load the checkpoint for the QMIX network
        checkpoint = os.path.join(f'last_networks/qmix/{self.message_type}', f'mixer')
        checkpoint = torch.load(checkpoint)
        self.qmix_net_eval.load_state_dict(checkpoint['network_state_dict'])  # Load QMIX network state
        self.mixing_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state for QMIX network
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())  # Copy current QMIX network weights to target network

    def load_network(self):
        """
        Args:
            None
            
        Returns:
            None
        
        Explanation:
            Loads the weights of the trained agent and QMIX networks from the saved checkpoint for inference or 
            continued training.
        """
        print('Loading Networks......')
        for number, network in enumerate(self.agent_networks):
            # Load the trained agent network weights from the saved checkpoint
            checkpoint = os.path.join(f'saved_networks/qmix/{self.message_type}', f'qmix_{number}')
            checkpoint = torch.load(checkpoint)
            network.load_state_dict(checkpoint['network_state_dict'])  # Load agent network weights
            self.target_agent_networks[number].load_state_dict(network.state_dict())  # Update target agent network
            self.agent_optimizers[number].load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state
        # Load the trained QMIX network weights
        checkpoint = os.path.join(f'saved_networks/qmix/{self.message_type}', f'mixer')
        checkpoint = torch.load(checkpoint)
        self.qmix_net_eval.load_state_dict(checkpoint['network_state_dict'])  # Load QMIX network weights
        self.mixing_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state for QMIX network
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())  # Copy current QMIX network weights to target network

    def init_hyperparams(self, ep_length, total_episodes):
        """
        Args:
            ep_length (int): The length of each episode.
            total_episodes (int): The total number of training episodes.
        
        Returns:
            None
        
        Explanation:
            Initializes the hyperparameters from the configuration file for the agent's learning process.
        """
        self.episode_length = ep_length 
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file) 
        self.gamma = float(params.get('gamma', 0.99)) 
        self.lr = float(params.get('lr', 2.5e-4)) 
        self.min_lr = float(params.get('min_lr', 2.5e-4)) # Standard = no lr annealing
        self.grad_norm_clip = float(params.get('grad_norm_clip', 0.5)) 
        self.start_epsilon = float(params.get('start_epsilon', 1))
        self.end_epsilon = float(params.get('end_epsilon', 0.01))
        self.start_temperature = float(params.get('start_temperature', 0.5))
        self.end_temperature = float(params.get('end_temperature', 0.01))
        self.fc = int(params.get('fc', 256)) 
        self.update_interval = int(params.get('update_interval', 10)) 
        self.message_type = params.get('message_type', 'simple') 
        self.exploration = params.get('exploration', 'greedy') 
        self.anneal_type = params.get('lr_anneal', 'linear') 
        self.training_steps = 0 
        self.decay_steps = total_episodes * 0.8 

    def epsilon_annealing(self):
        """
        Args:
            None
            
        Returns:
            float: The annealed epsilon value for exploration-exploitation trade-off.
        
        Explanation:
            Computes the epsilon value that controls exploration. It gradually decays from a starting value 
            to the final epsilon.
        """
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(-self.training_steps / self.decay_steps)
        return epsilon

    def anneal_lr(self):
        """
        Args: 
            steps: Current step or episode number
    
        Returns: None

        Explanation: Decrease the learning rate through the episodes 
                to promote exploitation over exploration.
        """
        steps = self.training_steps
        frac = (steps - 1) / self.decay_steps
    
        if self.anneal_type == "linear":
            # Linear annealing: Decrease the learning rate linearly
            new_lr = self.lr * (1 - frac)
        else:
            # Exponential annealing: Decrease the learning rate exponentially
            new_lr = self.lr * (self.min_lr / self.lr) ** frac
        
        # Ensure that learning rate does not go below the minimum learning rate
        new_lr = max(new_lr, self.min_lr)
    
        # Update the learning rates in the optimizers'
        for optimizer in self.agent_optimizers:
            optimizer.param_groups[0]["lr"] = new_lr
        self.mixing_optimizer.param_groups[0]["lr"] = new_lr
          

    def update_target_networks(self):
        """
        Explanation: Copies the weights from the current agent networks to the target networks for stable learning.
        """
        # Update target agent networks with the current agent networks' weights
        for i in range(self.n_agents):
            self.target_agent_networks[i].load_state_dict(self.agent_networks[i].state_dict()) 
        # Update target QMIX network with the current QMIX network's weights
        self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    def process_batch(self, batch):
        """
        Args:
            batch (dict): A batch of transitions containing states, actions, rewards, and other info
        
        Returns:
            q_total_eval (tensor): The evaluated Q-value of the batch.
            q_total_target (tensor): The target Q-value of the batch.
            rwrd (tensor): Rewards in the batch.
            term (tensor): Termination flags in the batch.
        
        Explanation: Processes a batch of data to compute the current and target Q-values for training.
        """
        state = batch['obs']  # Current state
        next_state = batch['obs_next']  # Next state
        # Concatenate the states to get a "central" state
        permuted_tensor = state.permute(1, 0, 2)  # Change the dimensions for batch processing
        central_state = permuted_tensor.reshape(self.episode_length, -1)  # Flatten the states to create central state
        permuted_tensor = next_state.permute(1, 0, 2)
        central_state_next = permuted_tensor.reshape(self.episode_length, -1)  # Flatten next state
        
        rwrd = batch['rewards']  # Rewards for the batch
        term = batch['dones']  # Termination flags for the batch
        
        # Get and transpose the actions taken by the agents
        actions = batch['actions'].long()  # Ensure actions are in long format
        transposed_tensor = actions.t()  # Transpose actions
        episode_actions = transposed_tensor  # Actions per agent
        
        # Get Q-values from agent networks based on current states and actions
        agent_qs = [agent(state[j]) for j, agent in enumerate(self.agent_networks)]
        agent_qs = torch.stack(agent_qs, dim=1)  # Stack the individual agent Q-values along the second dimension
        
        # Gather the Q-values for the chosen actions based on current states
        agent_qs = agent_qs.gather(2, episode_actions.unsqueeze(-1)).squeeze(-1)
        
        # Evaluate the Q-value for the central state using the QMIX network
        q_total_eval = self.qmix_net_eval(agent_qs, central_state)  # Compute Q-values using the evaluated QMIX network
        
        # Compute the target Q-values (for the next state) based on the target agent networks
        target_qs = [agent(next_state[j]) for j, agent in enumerate(self.target_agent_networks)]
        target_qs = torch.stack(target_qs, dim=1)  # Stack the target Q-values along the second dimension
        target_qs = target_qs.max(dim=-1)[0]  # Take the max value along the last dimension (optimal action)
        
        # Compute the target Q-value using the target QMIX network
        q_total_target = self.qmix_net_target(target_qs, central_state_next)
        
        return q_total_eval, q_total_target, rwrd, term  # Return evaluated Q-values, target Q-values, rewards, and terminations

    def train(self, batch, count):
        """
        Args:
            batch (list): A batch of transitions from the environment.
            count (int): The current step or episode number.

        Returns:
            pred_diff_arr (list): The array of predicted differences (TD-errors).
        
        Explanation: 
            Performs the training by computing the loss, applying backpropagation, 
            and updating the networks using the computed gradients.
        """
        total_episodes = len(batch)  # Total number of episodes in the current batch
        self.training_steps += 1  # Increment the training steps
        q_evals, q_targets, rewards, terminated = [], [], [], []  # Initialize lists to hold values during training
        pred_diff_arr = []  # Initialize list to hold the TD-errors
        
        for i in range(len(batch)):
            # Process each batch and get the Q-value evaluations and targets
            q_total_eval, q_total_target, rwrd, term = self.process_batch(batch[i])
            
            # Calculate the temporal difference (TD) error for the current episode
            pred_diff = rwrd[0] + torch.abs(q_total_eval - self.gamma * q_total_target)
            pred_diff = pred_diff.detach().numpy()  # Detach the result from the computation graph for further analysis
            aggregated_td_error = pred_diff.mean().item()  # Get the mean TD error for this batch
            pred_diff_arr.append(aggregated_td_error)  # Append the error to the array
            
            terminated.append(term)  # Append termination flags
            rewards.append(rwrd)  # Append rewards
            q_evals.append(q_total_eval)  # Append Q-value evaluations
            q_targets.append(q_total_target)  # Append target Q-values
        
        # Stack Q-values and rewards over all episodes in the batch
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        q_evals = q_evals.view(total_episodes, self.episode_length, 1)  # Reshape Q-values for each episode
        q_targets = q_targets.view(total_episodes, self.episode_length, 1)  # Reshape target Q-values for each episode
        rewards = torch.stack(rewards, dim=1)
        rewards = rewards[0].view(total_episodes, self.episode_length, 1)  # Reshape rewards
        dones = torch.stack(terminated, dim=1)
        dones = dones[0].view(total_episodes, self.episode_length, 1)  # Reshape termination flags
        
        # Compute the target values for Q-learning
        targets = rewards + self.gamma * q_targets * (1 - dones)
        # Compute the loss as the mean squared error between evaluated Q-values and target Q-values
        loss = F.mse_loss(q_evals, targets)
        
        # Zero the gradients for the optimizers
        self.mixing_optimizer.zero_grad()
        for opt in self.agent_optimizers:
            opt.zero_grad()
        
        loss.backward()  # Perform backpropagation to compute gradients
        
        # Clip gradients to avoid exploding gradients
        for agent in self.agent_networks:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        torch.nn.utils.clip_grad_norm_(self.qmix_net_eval.parameters(), self.grad_norm_clip)
        
        # Update the weights using the optimizer
        self.mixing_optimizer.step()  # Update QMIX network
        for opt in self.agent_optimizers:
            opt.step()  # Update each agent's network
        
        # Periodically update the target networks
        if count % self.update_interval == 0:
            self.update_target_networks()
        
        # Track the loss over time
        self.loss.append(loss.item())
        
        # Call the learning rate annealing function
        self.anneal_lr()  # Update learning rate based on the current step
        
        return pred_diff_arr  # Return the array of predicted differences (TD-errors)


    def temperature_annealing(self):
        """
        Explanation: Computes the annealed temperature for exploration, decaying over time to reduce randomness.
        """
        # Compute epsilon (temperature) using exponential decay
        epsilon = self.end_temperature + (self.start_temperature - self.end_temperature) * math.exp(-self.training_steps / self.decay_steps)
        return epsilon

    def eps_greedy(self, q_value, agent_idx):
        """
        Args:
            q_value (tensor): The Q-values for the current state.
            agent_idx (int): The index of the agent to select the action for.

        Returns:
            action (int): The chosen action based on the epsilon-greedy strategy.
        
        Explanation: 
            Chooses an action using an epsilon-greedy strategy based on the Q-values and exploration-exploitation trade-off.
        """
        epsilon = self.epsilon_annealing()  # Get epsilon value based on annealing
        random_value = random.random()  # Generate a random number for exploration
        # If a random value is smaller than epsilon, select a random action (exploration)
        if agent_idx == 4:
            if random_value < epsilon:
                action = random.randint(0, q_value.shape[0] - 1)  # Random action
            else:
                action = torch.argmax(q_value).item()  # Choose action with max Q-value (exploitation)
        else:
            if random_value < epsilon:
                action = random.randint(0, min(q_value.shape[0], 85) - 1)  # Random action
            else:
                action = torch.argmax(q_value[:85]).item()  # Choose action based on first 85 Q-values (exploitation)
        
        return action

    def bolzman_exploration(self, q_value, agent_idx):
        """
        Explanation: 
            Selects an action based on Boltzmann exploration, where the Q-values are transformed using a temperature parameter.
        """
        temperature = self.end_temperature  # Use end temperature for Boltzmann exploration
        soft = nn.Softmax(dim=-1)  # Softmax function to transform Q-values into probabilities
        
        # If agent index is 4, perform Boltzmann exploration over all Q-values
        if agent_idx == 4:
            prob = soft(q_value / temperature)  # Scale Q-values by temperature
            prob = prob.detach().numpy()  # Convert tensor to numpy for probability selection
            prob = prob / prob.sum()  # Normalize probabilities
        else:
            # Mask Q-values for agents other than 4 (set large negative values for unused actions)
            mask = np.ones_like(q_value.detach().numpy())
            mask[85:] = -np.inf  # Set a large negative value for actions > 85
            masked_q_value = q_value + torch.tensor(mask, dtype=torch.float32)
            prob = soft(masked_q_value / temperature)  # Apply softmax to masked Q-values
            prob = prob.detach().numpy()  # Convert tensor to numpy for probability selection
        
        # Select an action based on the computed probabilities
        action = np.random.choice(self.n_actions[agent_idx], p=prob)  # Randomly select action using probabilities
        
        return action

    def choose_actions(self, observations):
        """
        Args:
            observations (list): A list of observations from the environment.

        Returns:
            actions (list): A list of actions chosen by each agent.
            messages (list): A list of messages (if applicable) sent by each agent.
        
        Explanation: 
            Selects actions for each agent based on their observations and exploration strategy.
        """
        actions = []  # Initialize list for chosen actions
        messages = []  # Initialize list for messages (if any)
        
        # Iterate over each agent and choose an action
        for i, agent in enumerate(self.agent_networks):
            obs = observations[i]  # Get the observation for the current agent
            q_value = agent(torch.tensor(obs, dtype=torch.float32))  # Get Q-values for the current observation
            
            # Choose action based on exploration strategy (epsilon-greedy or Boltzmann)
            if self.exploration == 'greedy':
                action = self.eps_greedy(q_value, i)  # Choose action using epsilon-greedy
            else:
                action = self.bolzman_exploration(q_value, i)  # Choose action using Boltzmann exploration
            
            actions.append(action)  # Append the action to the list
            
            # If messages are enabled, prepare messages for communication
            if self.use_messages:
                message = self.message_handler[i].prepare_message(obs, action)
                messages.append(message)  # Append the message to the list
        
        return actions, messages  # Return the chosen actions and messages

  
