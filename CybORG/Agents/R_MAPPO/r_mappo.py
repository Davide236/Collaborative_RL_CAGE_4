# Import necessary libraries and modules
from CybORG.Agents.R_MAPPO.actor_network import ActorNetwork
from CybORG.Agents.R_MAPPO.buffer import ReplayBuffer
from CybORG.Agents.Messages.message_handler import MessageHandler
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import csv

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number, critic, critic_optim, messages):
        """
        Args:
            state_dimension: The dimension of the state space (number of features in the state).
            action_dimension: The dimension of the action space (number of possible actions).
            total_episodes: Total number of episodes for training.
            number: Agent number for identifying checkpoint and statistics files.
            critic: The critic network (value function).
            critic_optim: The optimizer for the critic network.
            messages: Whether to use messages for communication between agents.

        Returns:
            None

        Explanation:
            This constructor initializes the PPO (Proximal Policy Optimization) agent.
            It sets up hyperparameters, networks, optimizers, and initializes components like
            the replay buffer, message handler, and statistics tracking.
        """
        self.init_hyperparameters(total_episodes)  # Initialize hyperparameters for PPO
        self.memory = ReplayBuffer()  # Initialize the experience replay buffer for storing experiences
        self.init_checkpoint(number)  # Set up checkpoint files for saving/loading networks
        self.init_check_memory(number)  # Initialize CSV file for tracking statistics
        
        self.agent_number = number  # Store agent's number for checkpointing and statistics
        self.use_messages = messages  # Flag to indicate if messages will be used

        # Initialize actor network (policy) and its optimizer
        self.actor = ActorNetwork(state_dimension, action_dimension, self.hidden_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.eps)
        
        # Initialize critic network (value function) and its optimizer
        self.critic = critic
        self.critic_optimizer = critic_optim
        
        # Initialize message handler for agent communication (if enabled)
        self.message_handler = MessageHandler(message_type=self.message_type, number=self.agent_number)

    def get_action(self, state, state_value):
        """
        Args:
            state: The current observation state of the agent.
            state_value: The value function output, i.e., the expected return from the current state.

        Returns:
            action: The action chosen to be executed by the agent.
            message: An optional message sent along with the action (if communication is enabled).

        Explanation:
            This method determines the next action for the agent based on its current state.
            The actor network produces a distribution over possible actions, from which an action is sampled.
            If communication is enabled, the method also prepares a message to send along with the action.
        """
        # Normalize the state to ensure stable training by centering it around 0 and scaling to unit variance
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        state = torch.FloatTensor(normalized_state.reshape(1, -1))  # Convert the state into a tensor

        # Get the action distribution from the actor network
        action_distribution = self.actor(state)
        
        # Sample an action from the distribution
        action = action_distribution.sample()
        
        # Save the state, action, and value function to memory for later training
        self.memory.save_beginning_episode(state, action_distribution.log_prob(action).detach(), action.detach(), state_value.detach())
        
        # Prepare a message for communication if enabled
        message = []
        if self.use_messages:
            message = self.message_handler.prepare_message(state, action.item())

        return action.item(), message  # Return the action and the optional message

    def init_check_memory(self, number):
        """
        Args:
            number: The agent number used to create a unique statistics file for the agent.

        Returns:
            None

        Explanation:
            This method initializes lists to track statistics (entropy, critic loss, actor loss) during training.
            It also sets up the path to a CSV file where these statistics will be saved after training.
        """
        self.entropy = []  # List to track entropy (measure of randomness in the action distribution)
        self.critic_loss = []  # List to track the loss of the critic (value function)
        self.actor_loss = []  # List to track the loss of the actor (policy)
        self.save_path = f'saved_statistics/r_mappo/{self.message_type}/data_agent_{number}.csv'  # Path to save the statistics CSV

    def load_last_epoch(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            This method loads the last saved network and optimizer states from a checkpoint file.
            It is used to restore the agent's state when continuing from a previous training session.
        """
        print('Loading Last saved Networks and Optimizers......')
        checkpoint = torch.load(self.last_checkpoint_file_actor)  # Load checkpoint for actor network
        self.actor.load_state_dict(checkpoint['network_state_dict'])  # Load the actor's network parameters
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the actor optimizer parameters

    def load_network(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            This method loads the networks and optimizers from a checkpoint to initialize the agent's state.
        """
        print('Loading Networks and Optimizers......')
        checkpoint = torch.load(self.checkpoint_file_actor)  # Load the checkpoint for the actor network
        self.actor.load_state_dict(checkpoint['network_state_dict'])  # Load the actor's network weights
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the actor's optimizer state

    def init_checkpoint(self, number):
        """
        Args:
            number: The agent number used to generate unique checkpoint filenames.

        Returns:
            None

        Explanation:
            This method initializes file paths where the agent's model (actor) and its optimizer states will be saved.
        """
        # Define the paths for saving/loading checkpoints of the actor network
        self.checkpoint_file_actor = os.path.join(f'saved_networks/r_mappo/{self.message_type}', f'r_actor_mappo_{number}')
        self.last_checkpoint_file_actor = os.path.join(f'last_networks/r_mappo/{self.message_type}', f'r_actor_mappo_{number}')

    def save_statistics_csv(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            This method saves training statistics (entropy, critic loss, and actor loss) to a CSV file.
        """
        data = zip(self.entropy, self.critic_loss, self.actor_loss)  # Combine the statistics into a single iterable
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)  # Create a CSV writer
            writer.writerow(['Entropy', 'Critic Loss', 'Actor Loss'])  # Write header row
            writer.writerows(data)  # Write the tracked statistics to the CSV file

    def init_hyperparameters(self, episodes):
        """
        Args:
            episodes: The total number of episodes for training.

        Returns:
            None

        Explanation:
            This method initializes the hyperparameters for PPO from a YAML configuration file.
            It loads values such as learning rate, discount factor, number of epochs, etc.
        """
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')  # Path to the config file
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)  # Load hyperparameters from the YAML file
        # Extract individual hyperparameters from the config file
        self.epochs = int(params.get('epochs', 10))
        self.gamma = float(params.get('gamma', 0.99))
        self.clip = float(params.get('clip', 0.1))
        self.lr = float(params.get('lr', 2.5e-4))
        self.min_lr = float(params.get('min_lr', 5e-6))
        self.eps = float(params.get('eps', 1e-5))
        self.gae_lambda = float(params.get('gae_lambda', 0.95))
        self.entropy_coeff = float(params.get('entropy_coeff', 0.01))
        self.value_coefficient = float(params.get('value_coefficient', 0.5))
        self.max_grad_norm = float(params.get('max_grad_norm', 0.5))
        self.minibatch_number = int(params.get('minibatch_number', 1))
        self.hidden_size = int(params.get('hidden_size', 256))
        self.target_kl = float(params.get('target_kl', 0.02))
        self.message_type = params.get('message_type', 'action')
        self.anneal_type = params.get('lr_anneal', 'linear')
        self.max_episodes = episodes  # Store the total number of episodes for training

    def anneal_lr(self, steps):
        """
        Args:
            steps: The current step or episode number in training.

        Returns:
            None

        Explanation:
            This method decreases the learning rate over time to shift from exploration to exploitation.
            The learning rate is either linearly or exponentially annealed based on the chosen strategy.
        """
        frac = (steps - 1) / self.max_episodes  # Calculate the fraction of episodes completed
        if self.anneal_type == "linear":
            new_lr = self.lr * (1 - frac)  # Linearly decay the learning rate
        else:
            new_lr = self.lr * (self.min_lr / self.lr) ** frac  # Exponentially decay the learning rate

        # Ensure the learning rate does not go below the minimum value
        new_lr = max(new_lr, self.min_lr)
        
        # Update the learning rates for both actor and critic optimizers
        self.actor_optimizer.param_groups[0]["lr"] = new_lr
        self.critic_optimizer.param_groups[0]["lr"] = new_lr

    def evaluate(self, global_obs, observations, actions, sequence_length):
        """
        Args:
            global_obs: The global observation used to evaluate the state value.
            observations: The individual agent observations.
            actions: The actions taken by the agent.
            sequence_length: The length of the input sequences for batch processing.

        Returns:
            state_value: The value function output for the current state.
            log_probs: The log probability of the taken actions.
            entropy: The entropy of the action distribution.

        Explanation:
            This method computes the value of the current state using the critic network,
            the log probabilities of the taken actions, and the entropy of the action distribution.
        """
        action_distribution = self.actor(observations, sequence_length)  # Get action distribution from the actor network
        state_value = self.critic(global_obs, sequence_length).squeeze()  # Get state value from the critic
        log_probs = action_distribution.log_prob(actions)  # Compute log probability of the actions taken
        entropy = action_distribution.entropy()  # Compute entropy of the action distribution for exploration
        return state_value, log_probs, entropy  # Return state value, log probabilities, and entropy

    # Calculation of General Advantage Estimation
    def calculate_gae(self, rewards, values, terminated):
        """
        Args:
            rewards: A list of rewards obtained at each timestep during an episode.
            values: A list of estimated state values from the critic for each timestep.
            terminated: A list indicating whether each timestep was a terminal state (True/False).

        Returns:
            advantage_list: A padded tensor containing the Generalized Advantage Estimation (GAE)
                            for each timestep, computed for all episodes.

        Explanation:
            This function calculates the Generalized Advantage Estimation (GAE) for each timestep. 
            The GAE method helps in reducing the variance of policy gradients while maintaining bias.
            It uses rewards and state values to compute the advantage at each timestep.
            - If the episode is terminated, we don't need to discount future rewards.
            - The advantages are calculated by bootstrapping the future rewards based on the state value function.
        """
        batch_advantage = []  # List to store advantages for each episode
        count = 0
        # Start from the end since it’s easier to calculate the GAE backwards
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, terminated):
            count += 1
            advantages = []
            last_advantage = 0  # Initialize the last advantage as zero
            # Start from the last timestep and calculate the advantage backwards
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Temporal Difference (TD) error for timestep t (excluding the last one)
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    # For the last timestep, there’s no future value to discount
                    delta = ep_rews[t] - ep_vals[t]
                # Advantage calculation according to GAE formula
                advantage = delta + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                # Store the advantage for the next step calculation
                last_advantage = advantage
                advantages.append(advantage)

            batch_advantage.append(torch.cat(advantages, dim=0).squeeze())  # Store the advantages for the episode

        # Reverse the advantages for each episode and pad them to match the longest sequence
        reversed_tensor_array = [torch.flip(tensor, dims=[0]) for tensor in batch_advantage]
        advantage_list = pad_sequence(reversed_tensor_array, batch_first=True, padding_value=0)  # Padding the sequences
        return advantage_list  # Return the padded advantage tensor

    # Save the different loss parameters
    def save_data(self, entropy_loss, c_loss, a_loss):
        """
        Args:
            entropy_loss: The entropy loss computed during the policy update.
            c_loss: The critic loss (mean squared error between predicted state values and return).
            a_loss: The actor loss computed from the policy gradient.

        Returns:
            None

        Explanation:
            This method saves the entropy loss, critic loss, and actor loss values to 
            the corresponding lists for later use (e.g., for tracking training progress 
            or for statistical analysis).
        """
        self.entropy.append(entropy_loss.item())  # Append the entropy loss
        self.critic_loss.append(c_loss.item())  # Append the critic loss
        self.actor_loss.append(a_loss.item())  # Append the actor loss

    # In this case only 1 worker (no parallel implementation for easier debugging)
    def set_initial_state(self, workers):
        """
        Args:
            workers: The number of workers used for parallel computation (if applicable).

        Returns:
            None

        Explanation:
            This method sets the initial state of the actor network. Since only 1 worker is
            used in this case (for easier debugging), it initializes the actor network’s internal state.
        """
        self.actor.get_init_state(workers)  # Initialize the actor's hidden state


    def learn(self, total_steps):
        """
        Args:
            total_steps: The total number of training steps taken so far. This is used to adjust 
                         the learning rate decay.

        Returns:
            None

        Explanation:
            This function is the main learning process for the agent. It uses the experience collected
            from previous steps (observations, actions, rewards, etc.) to update the agent's policy.
            The steps followed include:
                1. Get a batch of experiences from memory.
                2. Apply reward scaling and compute advantages using GAE.
                3. Normalize the advantages.
                4. Update the learning rate based on the current training step.
                5. Perform policy updates for the specified number of epochs:
                    - Calculate the policy loss (actor loss) using the PPO clipped objective.
                    - Update the critic network by minimizing the MSE loss between predicted and target values.
                6. Save the loss values and clear memory.
        """
        # Extract necessary data (observations, actions, rewards, etc.) from the memory buffer
        obs, acts, logprob, state_values, global_obs, rewards, state_vals_unpadded, terminal = self.memory.get_batch()
        sequence_length = state_values.shape[1]  # The sequence length (timesteps per episode)
        steps = state_values.shape[0]  # The total number of steps (episodes)
        
        # Initialize losses to zero
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0
        
        # Calculate Generalized Advantage Estimation (GAE) for the batch of episodes
        A_k = self.calculate_gae(rewards, state_vals_unpadded, terminal)

        # Compute the return for each timestep (advantage + value function)
        rtgs = A_k + state_values.detach()
        
        # Normalize the advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)
        A_k = A_k.view(-1)  # Flatten the advantage tensor
        rtgs = rtgs.view(-1)  # Flatten the return tensor
        state_values = state_values.view(-1)  # Flatten the state values tensor

        # Anneal the learning rate based on the total training steps
        self.anneal_lr(total_steps)

        # Perform updates for the defined number of epochs
        for i in range(self.epochs):
            self.set_initial_state(steps)  # Initialize the actor's state for training
            self.critic.get_init_state(steps)  # Initialize the critic's state

            # Get mini-batches for training (one for each timestep in the episode)
            mini_obs = obs
            mini_acts = acts
            mini_log_prob = logprob
            mini_advantage = A_k
            mini_rtgs = rtgs
            
            # Evaluate the current policy and compute entropy, log probabilities, and state values
            mini_state_values, curr_log_probs, entropy = self.evaluate(global_obs, mini_obs, mini_acts, sequence_length)

            # Calculate entropy loss (a measure of exploration)
            entropy_loss = entropy.mean()

            # Compute the log ratio between current and old policies
            logrations = curr_log_probs - mini_log_prob
            ratios = torch.exp(logrations)

            # Approximate the KL divergence between the old and new policy
            approx_kl = ((ratios - 1) - logrations).mean()

            # Compute the actor loss using PPO objective (clipped version)
            actor_loss1 = ratios * mini_advantage
            actor_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
            actor_loss = (-torch.min(actor_loss1, actor_loss2)).mean()

            # Include entropy loss for better exploration
            actor_loss = actor_loss - entropy_loss * self.entropy_coeff

            # Critic loss: Mean Squared Error between predicted state values and actual returns
            critic_loss = nn.MSELoss()(mini_state_values, mini_rtgs)

            # Backpropagate the actor loss and optimize the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Gradient clipping to avoid exploding gradients
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Backpropagate the critic loss and optimize the critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Gradient clipping for the critic network
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # If the KL divergence exceeds the target, stop the training to prevent instability
            if approx_kl > self.target_kl:
                print(f"Breaking here: KL Divergence too large: {approx_kl}")
                break

        # Clear the memory after training the model
        self.memory.clear_rollout_memory()

        # Save the training results (entropy, actor loss, critic loss)
        self.save_data(entropy_loss, critic_loss, actor_loss)
