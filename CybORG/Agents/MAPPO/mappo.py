from CybORG.Agents.MAPPO.actor_network import ActorNetwork
from CybORG.Agents.MAPPO.buffer import ReplayBuffer
from CybORG.Agents.Messages.message_handler import MessageHandler
from torch.distributions import Categorical
import torch
import numpy as np
import torch.nn as nn
import yaml
import os

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number, critic, messages):
        """
        Args:
            state_dimension (int): The dimension of the input state space.
            action_dimension (int): The dimension of the output action space.
            total_episodes (int): Total number of episodes for training.
            number (int): Identifier for the agent (useful in multi-agent setups).
            critic (object): Critic network instance for value function approximation.
            messages (bool): Whether the agent uses messaging.

        Returns: 
            None

        Explanation:
            Initializes hyperparameters, memory buffers, and networks for the PPO agent.
        """
        self.init_hyperparameters(total_episodes)  # Load hyperparameters from configuration
        self.memory = ReplayBuffer()  # Initialize replay buffer for storing episodes
        self.init_checkpoint(number)  # Initialize checkpoint file paths
        self.init_check_memory(number)  # Initialize memory for tracking training statistics

        self.agent_number = number  # Agent identifier
        self.use_messages = messages  # Messaging toggle

        # Actor network for policy approximation
        self.actor = ActorNetwork(state_dimension, action_dimension, self.lr, self.eps, self.fc)
        
        # Critic network for value function approximation
        self.critic = critic  
        # Message handler for preparing and sending messages
        self.message_handler = MessageHandler(message_type=self.message_type, number=self.agent_number)
    
    def get_action(self, state, state_value):
        """
        Args:
            state (array-like): The current state observation of the agent.
            state_value (float): Value of the current state as predicted by the critic.

        Returns:
            tuple: Selected action and message (if messaging is enabled).

        Explanation:
            Takes the current state, normalizes it, and computes the action 
            using the actor network. Saves relevant data for training and 
            optionally prepares messages.
        """
        # Normalize the state to avoid large variance
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)
        state = torch.FloatTensor(normalized_state.reshape(1, -1))  # Reshape and convert to tensor

        # Select action based on the current policy
        action, logprob = self.actor.action_selection(state)
        # Save the state, action, log-probability, and state value to memory
        self.memory.save_beginning_episode(state, logprob, action, state_value)

        # Prepare a message if messaging is enabled
        message = []
        if self.use_messages:
            message = self.message_handler.prepare_message(state, action.item())

        return action.item(), message

    def init_check_memory(self, number):
        """
        Args:
            number (int): Identifier for the agent.

        Returns:
            None

        Explanation:
            Initializes arrays to store training metrics such as loss and entropy 
            for later analysis.
        """
        self.entropy = []  # List to store entropy values
        self.critic_loss = []  # List to store critic loss values
        self.actor_loss = []  # List to store actor loss values
        # Path to save training statistics as a CSV file
        self.save_path = f'saved_statistics/mappo/{self.message_type}/data_agent_{number}.csv'

    def load_last_epoch(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            Loads the actor network and its optimizer state from the most recent checkpoint.
        """
        print('Loading Last saved Networks and Optimizers......')
        checkpoint = torch.load(self.last_checkpoint_file_actor)  # Load checkpoint
        self.actor.load_state_dict(checkpoint['actor_state_dict'])  # Load actor network weights
        self.actor.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])  # Load optimizer state

    def load_network(self):
        """
        Args:
            None

        Returns:
            None

        Explanation:
            Loads the actor network and its optimizer state from the main checkpoint.
        """
        print('Loading Networks and Optimizers......')
        checkpoint = torch.load(self.checkpoint_file_actor)  # Load checkpoint
        self.actor.load_state_dict(checkpoint['network_state_dict'])  # Load actor network weights
        self.actor.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state

    def init_checkpoint(self, number):
        """
        Args:
            number (int): Identifier for the agent.

        Returns:
            None

        Explanation:
            Initializes file paths for saving and loading checkpoint files.
        """
        # Paths for saving and loading checkpoints
        self.checkpoint_file_actor = os.path.join(f'saved_networks/mappo/{self.message_type}', f'actor_mappo_{number}')
        self.last_checkpoint_file_actor = os.path.join(f'last_networks/mappo/{self.message_type}', f'actor_mappo_{number}')

    def init_hyperparameters(self, episodes):
        """
        Args:
            episodes (int): Total number of training episodes.

        Returns:
            None

        Explanation:
            Loads hyperparameters from a YAML configuration file or uses default values.
        """
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)

        # Hyperparameter values
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
        self.fc = int(params.get('fc', 256))
        self.target_kl = float(params.get('target_kl', 0.02))
        self.message_type = params.get('message_type', 'simple')
        self.max_episodes = episodes
        self.anneal_type = params.get('lr_anneal', 'linear')

    def anneal_lr(self, steps):
        """
        Args:
            steps (int): Current training step or episode.

        Returns:
            None

        Explanation:
            Adjusts the learning rate based on the current training step. Supports 
            linear and exponential annealing to encourage convergence over time.
        """
        # Fractional progress of training
        frac = (steps - 1) / self.max_episodes

        # Update learning rate based on annealing type
        if self.anneal_type == "linear":
            new_lr = self.lr * (1 - frac)
        else:  # Exponential annealing
            new_lr = self.lr * (self.min_lr / self.lr) ** frac

        # Ensure the learning rate does not drop below the minimum
        new_lr = max(new_lr, self.min_lr)

        # Update learning rates in the actor and critic optimizers
        self.actor.actor_optimizer.param_groups[0]["lr"] = new_lr
        self.critic.critic_optimizer.param_groups[0]["lr"] = new_lr

    def evaluate(self, global_obs, observations, actions):
        """
        Args: 
            global_obs (tensor): Global observations of the environment, used by the critic.
            observations (tensor): Local observations specific to the agent.
            actions (tensor): Actions performed corresponding to the observations.

        Returns: 
            state_value (tensor): Estimated state value from the critic network.
            log_probs (tensor): Logarithmic probabilities of the actions under the policy.
            entropy (tensor): Entropy of the action distribution, indicating randomness.

        Explanation: 
            This function uses the actor and critic networks to evaluate the state value
            (critic), the log-probabilities of actions (actor), and the entropy of the 
            action distribution. This information is critical for policy updates.
        """
        # Query the critic for state values based on global observations
        state_value = self.critic.get_state_value(global_obs).squeeze()

        # Compute action probabilities using the actor network
        masked_action_probs = self.actor(observations)
        # Create a categorical distribution over actions
        dist = Categorical(masked_action_probs)

        # Compute log probabilities and entropy for the actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return state_value, log_probs, entropy

    def calculate_gae(self, rewards, values, terminated):
        """
        Args: 
            rewards (list): Rewards achieved by the agent at each timestep, grouped by episodes.
            values (list): State values corresponding to each timestep, grouped by episodes.
            terminated (list): Termination flags indicating the end of episodes.

        Returns: 
            advantage_list (tensor): Calculated advantages for each timestep, as a tensor.

        Explanation:
            This function calculates the Generalized Advantage Estimation (GAE) for each timestep. 
            The GAE method helps in reducing the variance of policy gradients while maintaining bias.
            It uses rewards and state values to compute the advantage at each timestep.
            - If the episode is terminated, we don't need to discount future rewards.
            - The advantages are calculated by bootstrapping the future rewards based on the state value function.
        """
        batch_advantage = []
        # Iterate through each episode to compute advantages
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, terminated):
            advantages = []
            last_advantage = 0  # Initialize the last advantage as zero
            # Traverse the episode in reverse to calculate advantages
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Compute TD error for intermediate timesteps
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    # For the last timestep, exclude future rewards
                    delta = ep_rews[t] - ep_vals[t]
                # Generalized Advantage Estimation formula
                advantage = delta + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                # Insert the computed advantage at the beginning of the list
                advantages.insert(0, advantage)
            # Extend batch_advantage with episode advantages
            batch_advantage.extend(advantages)

        # Convert to tensor for compatibility with PyTorch operations
        advantage_list = torch.tensor(batch_advantage, dtype=torch.float)
        return advantage_list

    def save_data(self, entropy_loss, c_loss, a_loss):
        """
        Args: 
            entropy_loss (tensor): Entropy loss for the current training iteration.
            c_loss (tensor): Critic loss for the current training iteration.
            a_loss (tensor): Actor loss for the current training iteration.

        Returns: 
            None

        Explanation: 
            This function saves the entropy, critic, and actor losses into corresponding 
            lists for later analysis and debugging.
        """
        self.entropy.append(entropy_loss.item())
        self.critic_loss.append(c_loss.item())
        self.actor_loss.append(a_loss.item())

    def learn(self, total_steps):
        """
        Args:
            total_steps (int): The total number of training steps completed so far.
                            This is used to adjust the learning rate during training.

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
        # Retrieve training data from memory
        obs, global_obs, acts, logprob, rewards, state_vals, terminal = self.memory.get_batch()
        step = acts.size(0)  # Total number of timesteps in the batch
        index = np.arange(step)  # Index array for shuffling

        # Initialize losses
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0

        # Determine the size of minibatches
        minibatch_size = step // self.minibatch_number

        # Calculate advantages using GAE
        A_k = self.calculate_gae(rewards, state_vals, terminal)

        # Evaluate state values for global observations
        state_values, _, _ = self.evaluate(global_obs, obs, acts)

        # Calculate returns-to-go (RTG)
        rtgs = A_k + state_values.detach()

        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        # Training loop for multiple epochs
        for _ in range(self.epochs):
            self.anneal_lr(total_steps)  # Reduce learning rate
            np.random.shuffle(index)  # Shuffle indices for minibatches

            # Process minibatches
            for start in range(0, step, minibatch_size):
                end = start + minibatch_size
                idx = index[start:end]
                # Retrieve minibatch data
                mini_obs = obs[idx]
                mini_global_obs = global_obs[idx]
                mini_acts = acts[idx]
                mini_log_prob = logprob[idx]
                mini_advantage = A_k[idx]
                mini_rtgs = rtgs[idx]

                # Evaluate the current policy and compute losses
                state_values, curr_log_probs, entropy = self.evaluate(mini_global_obs, mini_obs, mini_acts)

                # Calculate policy loss
                entropy_loss = entropy.mean()
                logrations = curr_log_probs - mini_log_prob
                ratios = torch.exp(logrations)
                approx_kl = ((ratios - 1) - logrations).mean()
                actor_loss1 = ratios * mini_advantage
                actor_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                actor_loss = -torch.min(actor_loss1, actor_loss2).mean() - entropy_loss * self.entropy_coeff

                # Update actor network
                self.actor.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor.actor_optimizer.step()

                # Calculate critic loss and update critic network
                critic_loss = nn.MSELoss()(state_values, mini_rtgs)
                self.critic.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic.critic_optimizer.step()

            # Stop training early if KL divergence exceeds threshold
            if approx_kl > self.target_kl:
                print(f"Breaking Here: {approx_kl}")
                break

        # Clear memory and save the results
        self.memory.clear_rollout_memory()
        self.save_data(entropy_loss, critic_loss, actor_loss)
