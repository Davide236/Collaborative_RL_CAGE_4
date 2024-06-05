from CybORG.Agents.IPPO.networks import ActorCritic
from CybORG.Agents.Messages.message_handler import MessageHandler
from torch.distributions import Categorical
import torch 
import torch.nn as nn
import numpy as np

import os
import csv

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number, messages):
        # Initialize Hyperparameters, Rollout memory and Checkpoints
        self.init_hyperparameters(total_episodes)
        self.init_rollout_memory()
        self.init_checkpoint(number)
        self.init_check_memory(number)
        # Initialize actor and critic network
        self.policy = ActorCritic(state_dimension, action_dimension, self.lr, self.eps)
        self.use_messages = messages
        self.message_handler = MessageHandler()
        self.agent_number = number
    
    
    def get_action(self, state):
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
        message = []
        if self.use_messages:
            message = self.message_handler.extract_subnet_info(state, self.agent_number)
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Add small epsilon to avoid division by zero
        state = torch.FloatTensor(normalized_state.reshape(1,-1)) # Flatten the state
        action, logprob, state_value = self.policy.action_selection(state) # Under the old policy
        # Save state, log probability, action and state value to rollout memory
        self.observation_mem.append(state) 
        self.logprobs_mem.append(logprob)
        self.actions_mem.append(action) 
        self.episodic_state_val.append(state_value) 
        return action.item(), message
    
    # Initialize arrays to save important information for the training
    def init_check_memory(self, number):
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.save_path = f'saved_statistics\data_agent_{number}.csv'
    
    def save_last_epoch(self):
        print('Saving Networks.....')
        torch.save(self.policy.actor.state_dict(),self.last_checkpoint_file_actor)
        torch.save(self.policy.critic.state_dict(),self.last_checkpoint_file_critic)
    
    # Load the last saved networks
    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        self.policy.actor.load_state_dict(torch.load(self.last_checkpoint_file_actor))
        self.policy.critic.load_state_dict(torch.load(self.last_checkpoint_file_critic))

    # Save both actor and critic networks of the agent
    def save_network(self):
        print('Saving Networks.....')
        torch.save(self.policy.actor.state_dict(),self.checkpoint_file_actor)
        torch.save(self.policy.critic.state_dict(),self.checkpoint_file_critic)
    
    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        self.policy.actor.load_state_dict(torch.load(self.checkpoint_file_actor))
        self.policy.critic.load_state_dict(torch.load(self.checkpoint_file_critic))

    # Initialize checkpoint to save the different agents
    def init_checkpoint(self, number):
        self.checkpoint_file_actor = os.path.join('saved_networks', f'actor_ppo_{number}')
        self.checkpoint_file_critic = os.path.join('saved_networks', f'critic_ppo_{number}')
        self.last_checkpoint_file_actor = os.path.join('last_networks', f'actor_ppo_{number}')
        self.last_checkpoint_file_critic = os.path.join('last_networks', f'critic_ppo_{number}')

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
        self.clip = 0.1 # Clipping value: 0.2 is the value recommended by the paper.
        self.lr = 2.5e-4 # Learning rate of optimizer
        self.eps = 1e-5 # Epsilon value of optimizer to improve stability
        self.gae_lambda = 0.95 # General advantage estimation
        self.max_episodes = episodes
        self.entropy_coeff = 0.01 # Entropy coefficient
        self.value_coefficient = 0.5 # State value coeff for loss calculation
        self.max_grad_norm = 0.5 # Gradient clipping value
        self.minibatch_number = 1
        self.target_kl = 0.02 # 0.02 is also an option here

    # Initialize the rollout memory
    def init_rollout_memory(self):
        self.observation_mem = []
        self.actions_mem = []
        self.rewards_mem = []
        self.terminal_mem = [] 
        self.logprobs_mem = []
        self.state_val_mem = [] 
        self.action_mask_mem = [] #### 
        self.episodic_rewards = [] #
        self.episodic_termination = []
        self.episodic_state_val = []
    
    # Clear the rollout memory
    def clear_rollout_memory(self):
        del self.observation_mem[:]
        del self.actions_mem[:]
        del self.rewards_mem[:]
        del self.terminal_mem[:]
        del self.logprobs_mem[:]
        del self.state_val_mem[:]
        del self.action_mask_mem[:]
    
    # Clear the episodic memory
    def clear_episodic(self):
        del self.episodic_rewards[:]
        del self.episodic_termination[:]
        del self.episodic_state_val[:]
     
    
    def anneal_lr(self, steps):
        """
        Args: None

        Returns: None

        Explanation: Decrease the learning rate through the episodes 
                    to promote eploitation over exploration
        """
        frac = (steps-1)/self.max_episodes
        new_lr = self.lr * (1-frac)
        self.policy.actor_optimizer.param_groups[0]["lr"] = new_lr
        self.policy.critic_optimizer.param_groups[0]["lr"] = new_lr
    
    def evaluate(self, observations, actions):
        """
        Args: 
            observations: list of observation (states) recorded by the agent
            actions: list of actions performed for each of the observations
            action_mask: list of masked action values at each timestep

        Returns: 
            state_value: the value associated with the input observation, obtained by querying the critic network
            log_probs: log probability of the input actions given the distribution
            entropy: entropy value of the action probability distribution

        Explanation: In this function the Actor and Critic network are queried given
                    arrays of observation and action, in order to return the state value
                    of the observations (critic network), the logarithmic probability
                    of the actions (actor network) and the entropy of the action distribution.
        """
        state_value = self.policy.critic(observations).squeeze()
        #masked_action_probs = torch.tensor(action_mask, dtype=torch.float) * self.policy.actor(observations)
        mean = self.policy.actor(observations)
        dist = Categorical(mean)
        #dist = Categorical(masked_action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        # Compute logits from the actor network
        #logits = self.policy.actor(observations)
        # Apply action masking to the logits
        #masked_logits = torch.where(action_mask.bool(), logits, torch.tensor(-1e8))
        #masked_distribution = Categorical(logits=masked_logits)
        # Compute log probabilities of actions, considering action masking
        #log_probs = masked_distribution.log_prob(actions)
    
        # Compute entropy of the action distribution, considering action masking
        #entropy = masked_distribution.entropy()
        return state_value, log_probs, entropy
    
    def calculate_gae(self, rewards, values, terminated):
        """
        Args: 
            rewards: list containing all the rewards (divided by episode) achieved by the agent
            values:  list containing all the state values (divided by episode) encountered by the agent
            terminated: list of booleans containing the termination flag for all the episodes

        Returns: 
            advantage_list: list of advantages for each timestep

        Explanation: In this function we want to calculate the advantages by taking into account
                    not only the achieved rewards, but also the estimated state value at the specific
                    time step. The terminated flag is used to make sure that the agent knows when 
                    an episode ends and a new one starts.
        """
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
                advantages.insert(0,advantage)
            batch_advantage.extend(advantages)
        # Return list of advantages
        advantage_list = torch.tensor(batch_advantage, dtype=torch.float) 
        return advantage_list
    
    # Save the different loss parameters
    def save_data(self, entropy_loss, c_loss, a_loss):
        self.entropy.append(entropy_loss.item())
        self.critic_loss.append(c_loss.item())
        self.actor_loss.append(a_loss.item())


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
        obs = torch.cat(self.observation_mem, dim=0)
        acts = torch.tensor(self.actions_mem, dtype=torch.float)
        logprob = torch.tensor(self.logprobs_mem, dtype=torch.float).flatten()
        step = acts.size(0)
        index = np.arange(step)
        # Save Losses
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0
        # Calculate the size of the minibatches
        minibatch_size = step // self.minibatch_number
        # Calculate advantage per timestep
        A_k = self.calculate_gae(self.rewards_mem, self.state_val_mem, self.terminal_mem)
        state_values, _, _ = self.evaluate(obs, acts)
        # Future rewards based on advantage and state value
        rtgs = A_k + state_values.detach()
        # Normalize the advantage
        A_k = (A_k - A_k.mean())/(A_k.std() + 1e-8)
        # Perform the updates for X amount of epochs
        for _ in range(self.epochs):
            # Reduce the learning rate
            self.anneal_lr(total_steps)
            np.random.shuffle(index) # Shuffle the index
            # Process each minibatch
            for start in range(0, step, minibatch_size):
                # Create variables for minibatch data
                end = start + minibatch_size
                idx = index[start:end]
                mini_obs = obs[idx]
                mini_acts = acts[idx]
                mini_log_prob = logprob[idx]
                mini_advantage = A_k[idx]
                mini_rtgs = rtgs[idx]
                state_values, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
                # Compute policy loss with the formula
                entropy_loss = entropy.mean()
                logrations = curr_log_probs - mini_log_prob
                ratios = torch.exp(logrations)
                approx_kl = ((ratios - 1) - logrations).mean()
                actor_loss1 = ratios*mini_advantage
                actor_loss2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*mini_advantage
                actor_loss = (-torch.min(actor_loss1,actor_loss2)).mean()
                actor_loss = actor_loss - entropy_loss*self.entropy_coeff
                critic_loss = nn.MSELoss()(state_values, mini_rtgs)

                # Actor Update
                self.policy.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping for the networks (L2 Normalization)
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.policy.actor_optimizer.step()
                # Critic update
                self.policy.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Gradient clipping for the networks (L2 Normalization)
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.policy.critic_optimizer.step()

            # Check if the update was too large
            if approx_kl > self.target_kl:
                print(f"Breaking Here: {approx_kl}")
                break
        # Clear memory
        self.clear_rollout_memory()
        # Save last results
        self.save_data(entropy_loss, critic_loss, actor_loss)

