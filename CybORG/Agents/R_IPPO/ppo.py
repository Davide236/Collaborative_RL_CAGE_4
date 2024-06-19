from CybORG.Agents.R_IPPO.network import Actor, Critic
from CybORG.Agents.R_IPPO.buffer import ReplayBuffer
import torch 
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import os
import csv

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number):
        # Initialize Hyperparameters, Rollout memory and Checkpoints
        self.init_hyperparameters(total_episodes)
        self.memory = ReplayBuffer()
        self.init_check_memory(number)
        self.init_checkpoint(number)
        self.actor = Actor(state_dimension,action_dimension)
        self.critic = Critic(state_dimension)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def get_action(self, state):
        """
        Args:
            state: The current observation state of the agent.
            terminal: Current terminal value (1 or 0)

        Returns:
            action: The action chosen to be executed.

        Explanation: This function input a state (observation) of the agent
                    and outputs an action which is sampled from a categorical
                    probability distribution of all the possible actions given
                    the state.
        """
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Add small epsilon to avoid division by zero
        state = torch.FloatTensor(normalized_state.reshape(1,-1)) # Flatten the state
        # Query actor and critic networks
        action_distribution = self.actor(state)#, terminal) 
        state_value = self.critic(state)#, terminal)
        # Part 3 - Continued: Collect partial trajectories
        action = action_distribution.sample()
        self.memory.save_beginning_episode(state,action_distribution.log_prob(action).detach(), action.detach(),state_value.detach())
        return action.item()
    
    # Initialize arrays to save important information for the training
    def init_check_memory(self, number):
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.save_path = f'saved_statistics/data_agent_{number}.csv'
        
    # Save the statistics to a csv file
    def save_statistics_csv(self):
        data = zip(self.entropy, self.critic_loss, self.actor_loss)
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entropy', 'Critic Loss', 'Actor Loss'])  # Write header
            writer.writerows(data)
    
    def save_last_epoch(self):
        print('Saving Networks.....')
        torch.save(self.actor.state_dict(),self.last_checkpoint_file_actor)
        torch.save(self.critic.state_dict(),self.last_checkpoint_file_critic)
    
    # Load the last saved networks
    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        self.actor.load_state_dict(torch.load(self.last_checkpoint_file_actor))
        self.critic.load_state_dict(torch.load(self.last_checkpoint_file_critic))

    # Save both actor and critic networks of the agent
    def save_network(self):
        print('Saving Networks.....')
        torch.save(self.actor.state_dict(),self.checkpoint_file_actor)
        torch.save(self.critic.state_dict(),self.checkpoint_file_critic)
    
    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        self.actor.load_state_dict(torch.load(self.checkpoint_file_actor))
        self.critic.load_state_dict(torch.load(self.checkpoint_file_critic))

    # Initialize checkpoint to save the different agents
    def init_checkpoint(self, number):
        self.checkpoint_file_actor = os.path.join('saved_networks', f'r_actor_ppo_{number}')
        self.checkpoint_file_critic = os.path.join('saved_networks', f'r_critic_ppo_{number}')
        self.last_checkpoint_file_actor = os.path.join('last_networks', f'r_actor_ppo_{number}')
        self.last_checkpoint_file_critic = os.path.join('last_networks', f'r_critic_ppo_{number}')

    # Initialize hyperparameters
    def init_hyperparameters(self, episodes):
        self.epochs = 10
        self.gamma = 0.99 # Discount factor
        self.clip = 0.2 # Clipping value: 0.2 is the value recommended by the paper.
        self.lr = 2.5e-4 # Learning rate of optimizer
        self.eps = 1e-5 # Epsilon value of optimizer to improve stability
        self.gae_lambda = 0.95 # General advantage estimation
        self.max_episodes = episodes
        self.entropy_coeff = 0.02 # Entropy coefficient
        self.value_coefficient = 0.5 # State value coeff for loss calculation
        self.max_grad_norm = 0.5 # Gradient clipping value
        self.minibatch_number = 1
        self.target_kl = 0.02 # 0.02 is also an option here


    
    # In this case only 1 worker (no parallel implementation for easier debugging)
    def set_initial_state(self, workers):
        self.actor.get_init_state(workers)
        self.critic.get_init_state(workers)
        
    def anneal_lr(self, steps):
        """
        Args: None

        Returns: None

        Explanation: Decrease the learning rate through the episodes 
                    to promote eploitation over exploration
        """
        frac = (steps-1)/self.max_episodes
        new_lr = self.lr * (1-frac)
        self.actor_optimizer.param_groups[0]["lr"] = new_lr
        self.critic_optimizer.param_groups[0]["lr"] = new_lr
    
    # Function which performs an evaluation (with the latest hidden states) of a list of state values and actions probabilities
    # given a history of observations and actions saved during rollout
    def evaluate(self, observations, actions, sequence_length):
        action_distribution = self.actor(observations, sequence_length)
        state_value = self.critic(observations, sequence_length).squeeze()
        log_probs = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        return state_value, log_probs, entropy
    
    # Calculation of General Advantage Estimation
    def calculate_gae(self, rewards, values, terminated):
        batch_advantage = []
        count = 0
        # Start from the end since its easier calculation
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, terminated):
            count += 1
            advantages = []
            last_advantage = 0 
            # Start from last
            for t in reversed(range(len(ep_rews))):
                if t+1 < len(ep_rews):
                    # TD (Temporal Difference) error for timestep t
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1-ep_dones[t+1]) - ep_vals[t]
                else:
                    # In case this is the last timestep we dont have to add discount
                    delta = ep_rews[t] - ep_vals[t]
                # Following A_gae formula, he advantage at a step is delta + (gamma*lambda)*previous advantage
                advantage = delta + self.gamma*self.gae_lambda*(1-ep_dones[t])*last_advantage
                # Save the advantage to be used in the next calculation
                last_advantage = advantage
                #advantages.append(advantage.unsqueeze(0))
                advantages.append(advantage)
            # TODO: Change this
            batch_advantage.append(torch.cat(advantages, dim=0).squeeze())
        # Return list of advantages (padded)
        reversed_tensor_array = [torch.flip(tensor, dims=[0]) for tensor in batch_advantage]
        advantage_list = pad_sequence(reversed_tensor_array, batch_first=True, padding_value=0) 
        return advantage_list
    
    # Save the different loss parameters
    def save_data(self, entropy_loss, c_loss, a_loss):
        self.entropy.append(entropy_loss.item())
        self.critic_loss.append(c_loss.item())
        self.actor_loss.append(a_loss.item())
    
    
    # Step 4: Learning from past observations
    def learn(self,total_steps):
        # Pad memory
        obs, acts, logprob, state_values, rewards, state_vals_unpadded, terminal = self.memory.get_batch()
        # Number of episodes and length of each episode
        counter = state_values.shape[0]
        sequence_length = state_values.shape[1]
        # Save Losses
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0
        # Calculate advantage per timestep
        A_k = self.calculate_gae(rewards, state_vals_unpadded, terminal)
        # Future rewards based on advantage and state value
        rtgs = A_k + state_values.detach()
        # Normalize the advantage
        A_k = (A_k - A_k.mean())/(A_k.std() + 1e-8)

        # Change representation of data
        A_k = A_k.view(-1)
        rtgs = rtgs.view(-1)
        state_values = state_values.view(-1)

        # Reduce learning rate
        self.anneal_lr(total_steps)
        # Perform the updates for X amount of epochs
        for i in range(self.epochs):
            self.set_initial_state(counter)
            # Get the data of the episode
            mini_obs = obs
            mini_acts = acts
            mini_log_prob = logprob
            mini_advantage = A_k
            mini_rtgs = rtgs
            # Calculation of entropy, state values and action log probabilities for the episode
            mini_state_values, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts, sequence_length)
            # Step 5. Policy Loss
            # Compute policy loss with the formula
            # TODO HERE: Delete 0s (curr_log), (mini_log), (mini_adv) (mini_rtgs) (mini_state_val)
            zero_mask = (mini_log_prob == 0)
            curr_log_probs[zero_mask] = 0
            mini_advantage[zero_mask] = 0
            mini_rtgs[zero_mask] = 0
            mini_state_values[zero_mask] = 0

            entropy_loss = entropy.mean()
            logrations = curr_log_probs - mini_log_prob
            ratios = torch.exp(logrations)
            approx_kl = ((ratios - 1) - logrations).mean()
            # Add to the KL divergence (for each episode)
            actor_loss1 = ratios*mini_advantage
            actor_loss2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*mini_advantage
            actor_loss = (-torch.min(actor_loss1,actor_loss2)).mean()
            # Calculate Actor loss with formula
            actor_loss = actor_loss - entropy_loss*self.entropy_coeff
            # Critic loss following the formula
            critic_loss = nn.MSELoss()(mini_state_values, mini_rtgs)
            # With the total calculated loss for all the episode, propagate the loss through the graph
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Gradient clipping for the networks (L2 Normalization)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # Gradient clipping for the networks (L2 Normalization)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            # Check if the update was too large
            # If this is true, it means that the update is too large - Usually means that there is an error in implementation or parameters
            # TODO: Change this
            if approx_kl > self.target_kl:
                print(f"Breaking Here: {approx_kl}")
                break
        # Clear memory
        self.memory.clear_rollout_memory()
        # Save last results
        self.save_data(entropy_loss, critic_loss, actor_loss)

