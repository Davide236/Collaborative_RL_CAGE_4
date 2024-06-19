from CybORG.Agents.R_MAPPO.actor_network import ActorNetwork
from CybORG.Agents.R_MAPPO.buffer import ReplayBuffer
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

import torch 
import torch.nn as nn
import numpy as np

import os
import csv

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number, critic, critic_optim):
        # Initialize Hyperparameters, Rollout memory and Checkpoints
        self.init_hyperparameters(total_episodes)
        self.memory = ReplayBuffer()
        self.init_checkpoint(number)
        self.init_check_memory(number)
        # Save the number of the agent
        self.agent_number = number
        # Initialize actor and critic network
        self.actor = ActorNetwork(state_dimension, action_dimension)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # Initialize critic
        self.critic = critic
        self.critic_optimizer = critic_optim
    
    def get_action(self, state, state_value):
        """
        Args:
            state: The current observation state of the agent.

        Returns:
            action: The action chosen to be executed.

        Explanation: This function input a state (observation) of the agent
                    and outputs an action which is sampled from a categorical
                    probability distribution of all the possible actions given
                    the state.
        """
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Add small epsilon to avoid division by zero
        state = torch.FloatTensor(normalized_state.reshape(1,-1)) # Flatten the state
        action_distribution = self.actor(state)
        action = action_distribution.sample()
        self.memory.save_beginning_episode(state, action_distribution.log_prob(action).detach(), action.detach(), state_value.detach())
        return action.item()
    
    # Initialize arrays to save important information for the training
    def init_check_memory(self, number):
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.save_path = f'saved_statistics\data_agent_{number}.csv'
    
    def save_last_epoch(self):
        print('Saving Networks and Optimizers.....')
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, self.last_checkpoint_file_actor)
    
    def load_last_epoch(self):
        print('Loading Last saved Networks and Optimizers......')
        checkpoint = torch.load(self.last_checkpoint_file_actor)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def save_network(self):
        print('Saving Networks and Optimizers.....')
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, self.checkpoint_file_actor)
    
    def load_network(self):
        print('Loading Networks and Optimizers......')
        checkpoint = torch.load(self.checkpoint_file_actor)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def init_checkpoint(self, number):
        self.checkpoint_file_actor = os.path.join('saved_networks', f'r_actor_mappo_{number}')
        self.last_checkpoint_file_actor = os.path.join('last_networks', f'r_actor_mappo_{number}')

    # Save the statistics to a csv file
    def save_statistics_csv(self):
        data = zip(self.entropy, self.critic_loss, self.actor_loss)
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entropy', 'Critic Loss', 'Actor Loss'])  # Write header
            writer.writerows(data)

    # Initialize hyperparameters
    def init_hyperparameters(self, episodes):
        self.epochs = 10
        self.gamma = 0.99 # Discount factor
        self.clip = 0.2 # Clipping value: 0.2 is the value recommended by the paper.
        self.lr = 2.5e-4 # Learning rate of optimizer
        self.eps = 1e-5 # Epsilon value of optimizer to improve stability
        self.gae_lambda = 0.95 # General advantage estimation
        self.max_episodes = episodes
        self.entropy_coeff = 0.01 # Entropy coefficient
        self.value_coefficient = 0.5 # State value coeff for loss calculation
        self.max_grad_norm = 0.5 # Gradient clipping value
        self.minibatch_number = 1 
        self.target_kl = 0.02 # 0.02 is also an option here


     
    
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

    
    def evaluate(self, global_obs, observations, actions, sequence_length):
        action_distribution = self.actor(observations, sequence_length)
        state_value = self.critic(global_obs, sequence_length).squeeze()
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
    
    # In this case only 1 worker (no parallel implementation for easier debugging)
    def set_initial_state(self, workers):
        self.actor.get_init_state(workers)
    
    
    def learn(self,total_steps):
        """
        Args: 
            total_steps: total number of training 'steps' done so far. Variable used for the decay of the learning rate

        Returns: 
            None

        Explanation: This is the main learning function of the agents, in which all the previously saved
                    data (observations, actions, logarithmic probabilities etc.) are used for the policy update
                    as it described in the PPO update formula.
        """
        # Transform the observations, actions and log probability list into tensors
        obs, acts, logprob, state_values, global_obs, rewards, state_vals_unpadded, terminal = self.memory.get_batch()
        sequence_length = state_values.shape[1]
        steps = state_values.shape[0]
        # Save Losses
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0
        # Calculate advantage per timestep
        A_k = self.calculate_gae(rewards, state_vals_unpadded, terminal)
        # TODO: Check this
        # state_values = self.critic(global_obs, sequence_length).squeeze()
        # Future rewards based on advantage and state value
        rtgs = A_k + state_values.detach()
        # Normalize the advantage
        A_k = (A_k - A_k.mean())/(A_k.std() + 1e-8)
        ### TODO: Check those
        A_k = A_k.view(-1)
        rtgs = rtgs.view(-1)
        state_values = state_values.view(-1)
        # Perform the updates for X amount of epochs
        self.anneal_lr(total_steps)
        for i in range(self.epochs):
            # TODO: Check those
            self.set_initial_state(steps)
            self.critic.get_init_state(steps)
            #self.actor.recurrent_cell = (actor_hidden_states, actor_cell_states)
            #self.critic.recurrent_cell = (critic_hidden_states, critic_cell_states)
            # Get the data of the episode
            mini_obs = obs
            mini_acts = acts
            mini_log_prob = logprob
            mini_advantage = A_k
            mini_rtgs = rtgs
            # Calculation of entropy, state values and action log probabilities for the episode
            mini_state_values, curr_log_probs, entropy = self.evaluate(global_obs,mini_obs, mini_acts, sequence_length)
            # Step 5. Policy Loss

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