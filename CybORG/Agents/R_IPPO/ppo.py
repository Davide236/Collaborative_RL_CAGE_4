from CybORG.Agents.R_IPPO.network import Actor, Critic
from CybORG.Agents.R_IPPO.buffer import ReplayBuffer
from CybORG.Agents.Messages.message_handler import MessageHandler
import torch 
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import yaml
import os

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number, messages):
        # Initialize Hyperparameters, Rollout memory, and Checkpoints
        self.init_hyperparameters(total_episodes)  # Initializes training hyperparameters
        self.memory = ReplayBuffer()  # Initializes the replay buffer for storing transitions
        self.init_check_memory(number)  # Initializes memory for statistics storage
        self.init_checkpoint(number)  # Initializes checkpoint for saving the agent's networks
        self.agent_number = number  # Stores the agent's number
        self.use_messages = messages  # Flag to determine if message passing is used
        self.actor = Actor(state_dimension, action_dimension, self.hidden_size)  # Initializes the actor network
        self.critic = Critic(state_dimension, self.hidden_size)  # Initializes the critic network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)  # Optimizer for actor
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)  # Optimizer for critic
        self.message_handler = MessageHandler(message_type=self.message_type, number=self.agent_number)  # Initializes message handler
    
    def get_action(self, state):
        """
        Args:
            state: The current observation state of the agent.

        Returns:
            action: The action chosen to be executed.

        Explanation: 
            This function takes in the state (observation) of the agent, normalizes it, and
            feeds it into the actor network to get a probability distribution over possible actions.
            The action is then sampled from this distribution and returned.
        """
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Normalize the state
        state = torch.FloatTensor(normalized_state.reshape(1, -1))  # Convert state to tensor
        # Query actor and critic networks
        action_distribution = self.actor(state)  # Get action distribution from the actor network
        state_value = self.critic(state)  # Get state value from the critic network
        # Sample action from the distribution
        action = action_distribution.sample()  
        # Save the state, action, and log-probabilities for later use
        self.memory.save_beginning_episode(state, action_distribution.log_prob(action).detach(), action.detach(), state_value.detach())
        message = []
        if self.use_messages:
            message = self.message_handler.prepare_message(state, action.item())  # Prepare message if required
        return action.item(), message
    
    def init_check_memory(self, number):
        """
        Args:
            number: The agent's number to save the corresponding data.

        Explanation:
            This function initializes arrays for saving entropy, critic loss, and actor loss
            over the course of the agent's training. It also sets up a path to save the statistics.
        """
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.save_path = f'saved_statistics/r_ippo/{self.message_type}/data_agent_{number}.csv'
        
    def load_last_epoch(self):
        """
        Explanation:
            This function loads the latest saved checkpoint for both the actor and critic networks.
            The actor and critic network weights, as well as optimizer states, are restored from the checkpoint files.
        """
        print('Loading Last saved Networks......')
        actor_checkpoint = torch.load(self.last_checkpoint_file_actor)
        critic_checkpoint = torch.load(self.last_checkpoint_file_critic)
        self.actor.load_state_dict(actor_checkpoint['network_state_dict'])
        self.critic.load_state_dict(critic_checkpoint['network_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

    def load_network(self):
        """
        Explanation:
            This function loads both the actor and critic networks from the checkpoint files
            and restores the state of both the network weights and their optimizers.
        """
        print('Loading Networks......')
        actor_checkpoint = torch.load(self.checkpoint_file_actor)
        critic_checkpoint = torch.load(self.checkpoint_file_critic)
        self.actor.load_state_dict(actor_checkpoint['network_state_dict'])
        self.critic.load_state_dict(critic_checkpoint['network_state_dict'])
        self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

    def init_checkpoint(self, number):
        """
        Args:
            number: The agent's number to save the corresponding network files.

        Explanation:
            This function initializes the file paths for saving and loading checkpoint files
            for both the actor and critic networks.
        """
        self.checkpoint_file_actor = os.path.join(f'saved_networks/r_ippo/{self.message_type}', f'r_actor_ppo_{number}')
        self.checkpoint_file_critic = os.path.join(f'saved_networks/r_ippo/{self.message_type}', f'r_critic_ppo_{number}')
        self.last_checkpoint_file_actor = os.path.join(f'last_networks/r_ippo/{self.message_type}', f'r_actor_ppo_{number}')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/r_ippo/{self.message_type}', f'r_critic_ppo_{number}')

    def init_hyperparameters(self, episodes):
        """
        Args:
            episodes: The total number of episodes for training.

        Explanation:
            This function loads the hyperparameters from a YAML file, including values such as
            learning rate, gamma (discount factor), clip range, number of epochs, and others.
        """
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)  # Load hyperparameters from the YAML file
        self.epochs = int(params.get('epochs', 10))  # Number of epochs for training
        self.gamma = float(params.get('gamma', 0.99))  # Discount factor
        self.clip = float(params.get('clip', 0.1))  # Clipping factor for PPO
        self.lr = float(params.get('lr', 2.5e-4))  # Learning rate
        self.min_lr = float(params.get('min_lr', 5e-6))  # Minimum learning rate for annealing
        self.eps = float(params.get('eps', 1e-5))  # Epsilon value for numerical stability
        self.gae_lambda = float(params.get('gae_lambda', 0.95))  # GAE lambda for advantage estimation
        self.entropy_coeff = float(params.get('entropy_coeff', 0.01))  # Entropy coefficient for regularization
        self.value_coefficient = float(params.get('value_coefficient', 0.5))  # Value function coefficient
        self.max_grad_norm = float(params.get('max_grad_norm', 0.5))  # Max gradient norm for clipping
        self.minibatch_number = int(params.get('minibatch_number', 1))  # Number of minibatches
        self.hidden_size = int(params.get('hidden_size', 256))  # Hidden size of the networks
        self.target_kl = float(params.get('target_kl', 0.02))  # Target KL divergence for PPO
        self.message_type = params.get('message_type', 'action')  # Message type to be used for communication
        self.anneal_type = params.get('lr_anneal', 'linear')  # Type of learning rate annealing
        self.max_episodes = episodes  # Total number of episodes for training

    def set_initial_state(self, workers):
        """
        Args:
            workers: The number of workers to initialize for actor and critic networks.

        Explanation:
            This function initializes the state of the actor and critic networks, setting up 
            the networks for a given number of workers.
        """
        self.actor.get_init_state(workers)
        self.critic.get_init_state(workers)

    def anneal_lr(self, steps):
        """
        Args: 
            steps: Current step or episode number
    
        Returns: None

        Explanation: Decrease the learning rate through the episodes 
                    to promote exploitation over exploration. This helps in
                    reducing the exploration rate as the agent learns and stabilizes its policy.
        """
        # Calculate the fraction of the current step relative to the total steps (episodes)
        frac = (steps - 1) / self.max_episodes
    
        if self.anneal_type == "linear":
            # Linear annealing: Decrease the learning rate linearly
            new_lr = self.lr * (1 - frac)
        else:
            # Exponential annealing: Decrease the learning rate exponentially
            new_lr = self.lr * (self.min_lr / self.lr) ** frac
        
        # Ensure that the learning rate does not drop below the minimum learning rate
        new_lr = max(new_lr, self.min_lr)
    
        # Update the learning rates in both the actor and critic optimizers
        self.actor_optimizer.param_groups[0]["lr"] = new_lr
        self.critic_optimizer.param_groups[0]["lr"] = new_lr
        
    
    def evaluate(self, observations, actions, sequence_length):
        """
        Args:
            observations: A batch of input states.
            actions: A batch of actions taken by the agent.
            sequence_length: The length of the sequence of observations.

        Returns:
            state_value: The predicted state values for the batch of observations.
            log_probs: The log probabilities of the taken actions.
            entropy: The entropy of the action distribution (used for exploration).

        Explanation:
            This function queries both the Actor and Critic networks given the input observations and actions. 
            It computes:
                - The state value by passing the observations through the critic network.
                - The log probabilities of the actions given the policy (actor network).
                - The entropy of the action distribution to measure the randomness of the agent's actions.
        """
        action_distribution = self.actor(observations, sequence_length)  # Get action distribution from actor
        state_value = self.critic(observations, sequence_length).squeeze()  # Get predicted state values from critic
        log_probs = action_distribution.log_prob(actions)  # Compute log probabilities of taken actions
        entropy = action_distribution.entropy()  # Compute the entropy of the action distribution
        
        return state_value, log_probs, entropy  # Return values for use in loss calculation
    
    def calculate_gae(self, rewards, values, terminated):
        """
        Args:
            rewards: The rewards received at each time step in the episode.
            values: The predicted state values from the critic.
            terminated: A boolean list indicating whether each step is terminal.

        Returns:
            advantage_list: The computed Generalized Advantage Estimate (GAE) for each step.

        Explanation:
            This function calculates the Generalized Advantage Estimation (GAE) for each timestep. 
            The GAE method helps in reducing the variance of policy gradients while maintaining bias.
            It uses rewards and state values to compute the advantage at each timestep.
            - If the episode is terminated, we don't need to discount future rewards.
            - The advantages are calculated by bootstrapping the future rewards based on the state value function.
        """
        batch_advantage = []  # List to store advantages for each episode
        count = 0  # Counter to track episodes

        # Compute advantages for each episode in reverse order
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, terminated):
            count += 1
            advantages = []
            last_advantage = 0  # Initialize last advantage as 0
            
            # Iterate over the episode in reverse order to compute the advantage for each step
            for t in reversed(range(len(ep_rews))):
                if t+1 < len(ep_rews):
                    # Temporal Difference (TD) error for timestep t
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # For the last timestep, there is no future reward, so just subtract the value
                    delta = ep_rews[t] - ep_vals[t]
                
                # Compute advantage using GAE formula: A_t = delta + (gamma * lambda) * last_advantage
                advantage = delta + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage  # Update last advantage for next iteration
                advantages.append(advantage)  # Add the advantage for the current timestep
            
            # Reverse the advantages to match the original order of steps in the episode
            batch_advantage.append(torch.cat(advantages, dim=0).squeeze())

        # Reverse each tensor to get the correct order and pad sequences for batching
        reversed_tensor_array = [torch.flip(tensor, dims=[0]) for tensor in batch_advantage]
        advantage_list = pad_sequence(reversed_tensor_array, batch_first=True, padding_value=0)  # Pad sequences
        
        return advantage_list  # Return the padded advantages

    
    def save_data(self, entropy_loss, c_loss, a_loss):
        """
        Args:
            entropy_loss: The calculated entropy loss.
            c_loss: The critic loss (Mean Squared Error).
            a_loss: The actor loss (policy loss).

        Explanation:
            This function stores the computed loss values (entropy loss, critic loss, and actor loss) 
            to be used later for monitoring and debugging purposes.
            - The losses are appended to the corresponding lists (`entropy`, `critic_loss`, `actor_loss`).
        """
        self.entropy.append(entropy_loss.item())  # Append entropy loss to the list
        self.critic_loss.append(c_loss.item())  # Append critic loss to the list
        self.actor_loss.append(a_loss.item())  # Append actor loss to the list
    
    
    def learn(self, total_steps):
        """
        Args:
            total_steps: The total number of steps taken (or episodes).

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
                    - Use mini-batches for training to stabilize the updates.
                6. Save the loss values and clear memory.
        """
        # Retrieve a batch of observations and related data from memory
        obs, acts, logprob, state_values, rewards, state_vals_unpadded, terminal = self.memory.get_batch()
        
        # Number of episodes and the length of each episode
        counter = state_values.shape[0]  # Number of episodes
        sequence_length = state_values.shape[1]  # Length of each episode (sequence length)
        
        # Initialize loss values
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0
        
        # Calculate advantage using Generalized Advantage Estimation (GAE)
        A_k = self.calculate_gae(rewards, state_vals_unpadded, terminal)
        
        # Compute the return-to-go (RTG) as the advantage + state value
        rtgs = A_k + state_values.detach()
        
        # Normalize the advantage to have mean 0 and standard deviation 1
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        # Flatten the advantage and return-to-go for easy handling
        A_k = A_k.view(-1)
        rtgs = rtgs.view(-1)
        state_values = state_values.view(-1)

        # Anneal learning rate
        self.anneal_lr(total_steps)

        # Perform updates for a set number of epochs
        for i in range(self.epochs):
            self.set_initial_state(counter)  # Reset states to the initial values
            # Get mini-batches of data for the current episode
            mini_obs = obs
            mini_acts = acts
            mini_log_prob = logprob
            mini_advantage = A_k
            mini_rtgs = rtgs
            
            # Evaluate the current policy (get state values, log probabilities, and entropy)
            mini_state_values, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts, sequence_length)

            # Policy loss computation
            zero_mask = (mini_log_prob == 0)  # Mask out zero log probabilities to avoid computation errors
            curr_log_probs[zero_mask] = 0  # Replace zero log probabilities with zero
            mini_advantage[zero_mask] = 0  # Set corresponding advantages to zero
            mini_rtgs[zero_mask] = 0  # Set corresponding return-to-go values to zero
            mini_state_values[zero_mask] = 0  # Set state values to zero where log probs were zero

            # Compute entropy loss (regularization for exploration)
            entropy_loss = entropy.mean()
            
            # Compute the log ratios and the corresponding ratios
            logrations = curr_log_probs - mini_log_prob
            ratios = torch.exp(logrations)  # Importance sampling ratio
            approx_kl = ((ratios - 1) - logrations).mean()  # Approximate KL divergence
            
            # Calculate actor loss using the clipped objective function
            actor_loss1 = ratios * mini_advantage
            actor_loss2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
            actor_loss = (-torch.min(actor_loss1, actor_loss2)).mean()
            
            # Include entropy regularization in the actor loss
            actor_loss = actor_loss - entropy_loss * self.entropy_coeff

            # Critic loss using Mean Squared Error between predicted and actual returns
            critic_loss = nn.MSELoss()(mini_state_values, mini_rtgs)

            # Perform backpropagation for the actor and critic networks
            self.actor_optimizer.zero_grad()  # Zero the gradients for actor optimizer
            actor_loss.backward()  # Backpropagate actor loss
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # Clip gradients to prevent exploding gradients
            self.actor_optimizer.step()  # Step the actor optimizer
            
            self.critic_optimizer.zero_grad()  # Zero the gradients for critic optimizer
            critic_loss.backward()  # Backpropagate critic loss
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)  # Clip gradients for critic
            self.critic_optimizer.step()  # Step the critic optimizer

            # Check if the KL divergence exceeds the target threshold (early stopping condition)
            if approx_kl > self.target_kl:
                print(f"Breaking Here: {approx_kl}")  # If true, break the loop early
                break
        
        # Clear the memory after learning
        self.memory.clear_rollout_memory()
        
        # Save the last recorded losses for later analysis
        self.save_data(entropy_loss, critic_loss, actor_loss)

