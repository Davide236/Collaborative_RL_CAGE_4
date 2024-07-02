from CybORG.Agents.IPPO.networks import ActorCritic
from CybORG.Agents.Messages.message_handler import MessageHandler
from CybORG.Agents.IPPO.buffer import ReplayBuffer
from torch.distributions import Categorical
import torch 
import torch.nn as nn
import numpy as np
import yaml
import os
import csv

class PPO:
    def __init__(self, state_dimension, action_dimension, total_episodes, number, messages):
        # Initialize Hyperparameters, Rollout memory and Checkpoints
        self.init_hyperparameters(total_episodes)
        self.memory = ReplayBuffer()
        self.init_checkpoint(number)
        self.init_check_memory(number)
        # Initialize actor and critic network
        self.policy = ActorCritic(state_dimension, action_dimension, self.lr, self.eps, self.fc)
        self.use_messages = messages
        self.agent_number = number
        self.message_handler = MessageHandler(message_type=self.message_type, number=self.agent_number)
    
    
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
        normalized_state = (state - np.mean(state)) / (np.std(state) + 1e-8)  # Add small epsilon to avoid division by zero
        final_state = torch.FloatTensor(normalized_state.reshape(1,-1)) # Flatten the state
        action, logprob, state_value = self.policy.action_selection(final_state) # Under the old policy
        # Save state, log probability, action and state value to rollout memory
        self.memory.save_beginning_episode(final_state, logprob, action, state_value)
        message = []
        if self.use_messages:
            message = self.message_handler.prepare_message(state, action.item())
        return action.item(), message
    
    # Initialize arrays to save important information for the training
    def init_check_memory(self, number):
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.save_path = f'saved_statistics\ippo\{self.message_type}\data_agent_{number}.csv'
    
    # Load the last saved networks
    def load_last_epoch(self):
        print('Loading Last saved Networks......')
        # self.policy.actor.load_state_dict(torch.load(self.last_checkpoint_file_actor['network_state_dict']))
        # self.policy.critic.load_state_dict(torch.load(self.last_checkpoint_file_critic['network_state_dict']))
        # self.policy.actor_optimizer.load_state_dict(torch.load(self.last_checkpoint_file_actor['optimizer_state_dict']))
        # self.policy.critic_optimizer.load_state_dict(torch.load(self.last_checkpoint_file_critic['optimizer_state_dict']))
        self.policy.actor.load_state_dict(torch.load(self.last_checkpoint_file_actor))
        self.policy.critic.load_state_dict(torch.load(self.last_checkpoint_file_critic))



    # Load both actor and critic network of the agent
    def load_network(self):
        print('Loading Networks......')
        self.policy.actor.load_state_dict(torch.load(self.checkpoint_file_actor['network_state_dict']))
        self.policy.critic.load_state_dict(torch.load(self.checkpoint_file_critic['network_state_dict']))
        self.policy.actor_optimizer.load_state_dict(torch.load(self.checkpoint_file_actor['optimizer_state_dict']))
        self.policy.critic_optimizer.load_state_dict(torch.load(self.checkpoint_file_critic['optimizer_state_dict']))

    # Initialize checkpoint to save the different agents
    def init_checkpoint(self, number):
        self.checkpoint_file_actor = os.path.join(f'saved_networks/ippo/{self.message_type}', f'actor_ppo_{number}')
        self.checkpoint_file_critic = os.path.join(f'saved_networks/ippo/{self.message_type}', f'critic_ppo_{number}')
        self.last_checkpoint_file_actor = os.path.join(f'last_networks/ippo/{self.message_type}', f'actor_ppo_{number}')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/ippo/{self.message_type}', f'critic_ppo_{number}')

    # Initialize hyperparameters
    def init_hyperparameters(self, episodes):
        self.max_episodes = episodes
        config_file_path = os.path.join(os.path.dirname(__file__), 'hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        self.epochs = int(params.get('epochs', 10))
        self.gamma = float(params.get('gamma', 0.99))
        self.clip = float(params.get('clip', 0.1))
        self.lr = float(params.get('lr', 2.5e-4))
        self.eps = float(params.get('eps', 1e-5))
        self.gae_lambda = float(params.get('gae_lambda', 0.95))
        self.entropy_coeff = float(params.get('entropy_coeff', 0.01))
        self.value_coefficient = float(params.get('value_coefficient', 0.5))
        self.max_grad_norm = float(params.get('max_grad_norm', 0.5))
        self.minibatch_number = int(params.get('minibatch_number', 1))
        self.fc = int(params.get('fc', 256))
        self.target_kl = float(params.get('target_kl', 0.02))
        self.message_type = params.get('message_type', 'simple')

     
    
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
        mean = self.policy.actor(observations)
        dist = Categorical(mean)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
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
        obs, acts, logprob, rewards, state_val, terminal = self.memory.get_batch()
        step = acts.size(0)
        index = np.arange(step)
        # Save Losses
        critic_loss = 0
        actor_loss = 0
        entropy_loss = 0
        # Calculate the size of the minibatches
        minibatch_size = step // self.minibatch_number
        # Calculate advantage per timestep
        A_k = self.calculate_gae(rewards, state_val, terminal)
        state_values, _, _ = self.evaluate(obs, acts)
        # Future rewards based on advantage and state value
        rtgs = A_k + state_values.detach()
        # Normalize the advantage
        #A_k = (A_k - A_k.mean())/(A_k.std() + 1e-8)
        # Reduce the learning rate
        self.anneal_lr(total_steps)
        # Perform the updates for X amount of epochs
        for _ in range(self.epochs):
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
        self.memory.clear_rollout_memory()
        # Save last results
        self.save_data(entropy_loss, critic_loss, actor_loss)

