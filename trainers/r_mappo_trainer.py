from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.R_MAPPO.r_mappo import PPO
from CybORG.Agents.R_MAPPO.critic_network import CriticNetwork
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean
import numpy as np
import torch
import yaml
import os
from utils import save_statistics, save_agent_data_ppo, save_agent_network, RewardNormalizer


# Trainer Class for the Recurrent version of the MAPPO algorithm
class RecurrentMAPPOTrainer:
    # Standard length of CybORG episode
    EPISODE_LENGTH = 500

    def __init__(self, args):
        self.env = None  # CybORG simulation environment
        self.agents = None  # Dictionary of agents
        self.total_rewards = []  # Total rewards for all episodes
        self.partial_rewards = 0  # Accumulated rewards for the current batch
        self.best_reward = -8000  # Best reward achieved during training
        self.average_rewards = []  # Average rewards across batches
        self.count = 0  # Total step count across all episodes
        self.best_critic = None  # Path to the best critic checkpoint
        self.last_critic = None  # Path to the last critic checkpoint
        self.load_last_network = args.Load_last  # Flag to load the most recent network
        self.load_best_network = args.Load_best  # Flag to load the best-performing network
        self.messages = args.Messages  # Whether to enable inter-agent messaging
        self.rollout = args.Rollout  # Number of episodes between policy updates
        self.max_eps = args.Episodes  # Total number of episodes for training
        self.centralized_critic = None  # Centralized critic network
        self.critic_optimizer = None  # Optimizer for the centralized critic

    # Concatenate individual observations to create a global observation for the centralized critic
    def concatenate_observations(self, observations):
        observation_list = []  # Collect all agent observations
        for agent_name in self.agents.keys():
            observation_list.extend(observations[agent_name])
        normalized_state = (observation_list - np.mean(observation_list)) / (np.std(observation_list) + 1e-8)
        state = torch.FloatTensor(normalized_state.reshape(1, -1))  # Convert to PyTorch tensor
        return state
    
    # Initialize the centralized critic network for all agents
    def initialize_critic(self, env):
        # Load hyperparameters from YAML configuration file
        config_file_path = os.path.join(os.path.dirname(__file__), '../CybORG/Agents/MAPPO/hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        lr = float(params.get('lr', 2.5e-4))  # Learning rate
        eps = float(params.get('eps', 1e-5))  # Epsilon for optimizer
        fc = int(params.get('fc', 256))  # Fully connected layer size
        # Instantiate the centralized critic network
        centralized_critic = CriticNetwork(
            env.observation_space('blue_agent_4').shape[0],  # Observation space size
            env.observation_space('blue_agent_0').shape[0],  # Shared state space size
            5,  # Number of agents
            fc  # Number of units in hidden layers
        )
        # Create an optimizer for the critic
        critic_optimizer = torch.optim.Adam(centralized_critic.parameters(), lr=lr, eps=eps)
        message_type = params.get('message_type', 'simple')  # Messaging type configuration
        return centralized_critic, critic_optimizer, message_type

    # Initialize file paths for saving critic network checkpoints
    def init_checkpoint(self):
        checkpoint_file_critic = os.path.join('saved_networks', f'r_critic_mappoppo_central')
        last_checkpoint_file_critic = os.path.join('last_networks', f'r_critic_mappo_central')
        return checkpoint_file_critic, last_checkpoint_file_critic
    
    # Setup agents for MAPPO with PPO algorithms
    def setup_agents(self, env):
        agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],  # Observation space size
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),  # Action space size
                self.max_eps * self.EPISODE_LENGTH,  # Total timesteps
                agent,  # Agent identifier
                self.centralized_critic,  # Centralized critic network
                self.critic_optimizer,  # Critic optimizer
                self.messages  # Messaging flag
            )
            for agent in range(5)  # Number of agents
        }
        return agents

    # Initialize the CybORG environment and agents
    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,  # Blue agent class
            green_agent_class=EnterpriseGreenAgent,  # Green agent class
            red_agent_class=FiniteStateRedAgent,  # Red agent class
            steps=self.EPISODE_LENGTH  # Episode length
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Create CybORG environment with seed for reproducibility
        self.env = BlueFlatWrapper(env=cyborg)  # Wrap the environment for PPO compatibility
        self.env.reset()  # Reset the environment
        
        # Initialize critic, checkpoint paths, and agents
        self.best_critic, self.last_critic = self.init_checkpoint()
        self.centralized_critic, self.critic_optimizer, self.message_type = self.initialize_critic(self.env)
        self.checkpoint_critic = os.path.join(f'saved_networks/r_mappo/{self.message_type}', f'critic_ppo_central')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/r_mappo/{self.message_type}', f'critic_ppo_central')
        self.agents = self.setup_agents(self.env)
        print(f'Using agents {self.agents}')

        # Load pre-trained networks if specified
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
            self.centralized_critic.load_state_dict(torch.load(self.best_critic['network_state_dict']))
            self.critic_optimizer.load_state_dict(torch.load(self.best_critic['optimizer_state_dict']))
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()
            self.centralized_critic.load_state_dict(torch.load(self.last_critic['network_state_dict']))
            self.critic_optimizer.load_state_dict(torch.load(self.last_critic['optimizer_state_dict']))
            
    # Run the MAPPO training loop
    def run(self):
        self.initialize_environment()  # Initialize the environment and agents
        reward_normalizer = RewardNormalizer()  # Create a reward normalizer to stabilize training

        # Main training loop, iterating over episodes
        for i in range(self.max_eps):  # Iterate over the total number of episodes
            observations, _ = self.env.reset()  # Reset the environment at the start of each episode
            
            # Reset the initial state for each agent and the centralized critic at the start of each episode
            for agent_name, agent in self.agents.items():
                agent.set_initial_state(1)  # Reset agent's state
            self.centralized_critic.get_init_state(1)  # Reset the centralized critic's state

            r = []  # List to accumulate rewards for the current episode
            for j in range(self.EPISODE_LENGTH):  # Loop through each time step in the episode
                self.count += 1  # Increment the global step counter

                # Concatenate individual agent observations to form a global observation
                observations_list = self.concatenate_observations(observations)
                state_value = self.centralized_critic(observations_list)  # Get the value of the current state from the critic
                
                # Action selection for all agents (choose actions based on the observation and state value)
                actions_messages = {
                    agent_name: agent.get_action(observations[agent_name], state_value)  # Get action for each agent
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents  # Ensure the agent is part of the environment
                }
                # Separate actions and messages from the actions_messages tuple
                actions = {agent_name: action for agent_name, (action, _) in actions_messages.items()}
                messages = {agent_name: message for agent_name, (_, message) in actions_messages.items()}

                # Perform the selected actions in the environment
                if self.messages:
                    # If messaging is enabled, pass the messages to the environment along with actions
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    # If messaging is not enabled, just perform the actions
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Store rewards and the termination status for each agent in the current episode
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]  # Check if the agent is done (either terminated or truncated)
                    agent.memory.save_end_episode(reward_normalizer.normalize(reward[agent_name]), done, observations_list)

                # Check if all agents are done (either terminated or truncated)
                done = {agent: termination.get(agent, False) or truncation.get(agent, False) for agent in self.env.agents}
                if all(done.values()):  # If all agents are done, break the loop
                    break
                r.append(mean(reward.values()))  # Add the mean of the rewards for this timestep

            # After finishing the episode, update cumulative rewards
            self.partial_rewards += sum(r)  # Accumulate rewards for this batch
            self.total_rewards.append(sum(r))  # Add the total reward for the episode to the list
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {mean(self.total_rewards)}")  # Print stats for the episode

            # Perform policy update after a specific number of episodes (rollout)
            if (i + 1) % self.rollout == 0:
                avg_rwd = self.partial_rewards / self.rollout  # Calculate the average reward for the current batch
                self.average_rewards.append(avg_rwd)  # Store average reward for later analysis
                print(f"Average reward obtained before update: {avg_rwd}")

                # If the average reward exceeds the best reward, save the current agent and critic networks
                if avg_rwd > self.best_reward:
                    self.best_reward = avg_rwd  # Update the best reward achieved
                    for agent_name, agent in self.agents.items():
                        save_agent_network(agent.actor, agent.actor_optimizer, agent.checkpoint_file_actor)  # Save the agent's actor network
                    save_agent_network(self.centralized_critic, self.critic_optimizer, self.checkpoint_critic)  # Save the centralized critic network

                self.partial_rewards = 0  # Reset partial rewards for the next batch

            # Append the episodic memory for each agent and perform policy update after each rollout
            for agent_name, agent in self.agents.items():
                agent.memory.append_episodic()  # Append current episode to the agent's memory
                if (i + 1) % self.rollout == 0:  # Update policy after each rollout
                    print(f"Policy update for {agent_name}. Total steps: {self.count}")  # Print policy update status
                    agent.learn(self.count)  # Update the agent's policy using the collected experience

        # After all episodes are completed, save all collected training data
        save_agent_data_ppo(self.agents)  # Save the training data for PPO agents
        for agent_name, agent in self.agents.items():
            save_agent_network(agent.actor, agent.actor_optimizer, agent.last_checkpoint_file_actor)  # Save the agent's last network checkpoint
        save_agent_network(self.centralized_critic, self.critic_optimizer, self.last_checkpoint_file_critic)  # Save the last checkpoint of the centralized critic
        save_statistics(self.total_rewards, self.average_rewards)  # Save statistics of total rewards and average rewards over episodes
