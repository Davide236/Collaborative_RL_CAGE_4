# Import necessary libraries and modules
from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.MAPPO.mappo import PPO
from CybORG.Agents.MAPPO.critic_network import CriticNetwork
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean
import numpy as np
import torch
import os
import yaml
from utils import save_statistics, save_agent_data_ppo, save_agent_network, RewardNormalizer

# Trainer class for Multi-Agent PPO (MAPPO) algorithm
class MAPPOTrainer:
    # Standard episode length for CybORG environment
    EPISODE_LENGTH = 500

    def __init__(self, args):
        # Initialize trainer parameters
        self.env = None
        self.agents = None
        self.centralized_critic = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Total episodes count
        self.best_reward = -7000  # Track the best reward achieved
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.rollout = args.Rollout  # Number of episodes before policy update
        self.max_eps = args.Episodes  # Maximum number of training episodes

    @staticmethod
    def concatenate_observations(observations, agents):
        """
        Concatenates agent observations into a single global state.
        Includes specific handling of message data and mission phase.
        """
        observation_list = []
        messages_1_4 = []
        messages_0 = []
        mission_phase = 0

        for agent_name, agent in agents.items():
            if agent_name == 'blue_agent_0':
                mission_phase = observations[agent_name][0]
                messages_1_4 = observations[agent_name][-32:]
            elif agent_name == 'blue_agent_1':
                message_chunk = observations[agent_name][-32:]
                messages_0 = message_chunk[:8]
            observation_list.extend(observations[agent_name][1:-32])
        
        # Add mission phase and messages
        observation_list.insert(0, mission_phase)
        observation_list.extend(messages_0)
        observation_list.extend(messages_1_4)
        
        # Normalize the observation array
        normalized_state = (observation_list - np.mean(observation_list)) / (np.std(observation_list) + 1e-8)
        state = torch.FloatTensor(normalized_state.reshape(1, -1))
        return state

    @staticmethod
    def initialize_critic(env):
        """
        Initialize the centralized critic network.
        Parameters are loaded from a YAML configuration file.
        """
        config_file_path = os.path.join(os.path.dirname(__file__), '../CybORG/Agents/MAPPO/hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        
        # Extract hyperparameters
        lr = float(params.get('lr', 2.5e-4))
        eps = float(params.get('eps', 1e-5))
        fc = int(params.get('fc', 256))
        global_state = params.get('global_state', 'standard')

        # Calculate state dimension
        if global_state == 'standard':
            state_dim = env.observation_space('blue_agent_4').shape[0] + (
                env.observation_space('blue_agent_0').shape[0] * (5 - 1)
            )
        else:
            state_dim = 454  # Modify this as required

        # Initialize critic network
        centralized_critic = CriticNetwork(state_dim, lr, eps, fc)
        message_type = params.get('message_type', 'simple')
        return centralized_critic, message_type

    def initialize_environment(self):
        """
        Setup the CybORG environment, centralized critic, and PPO agents.
        """
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add seed for reproducibility
        env = BlueFlatWrapper(env=cyborg)
        env.reset()
        self.env = env

        # Initialize centralized critic
        self.centralized_critic, self.message_type = self.initialize_critic(env)
        self.checkpoint_critic = os.path.join(f'saved_networks/mappo/{self.message_type}', f'critic_ppo_central')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/mappo/{self.message_type}', f'critic_ppo_central')

        # Initialize PPO agents for each blue agent
        self.agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                self.max_eps * self.EPISODE_LENGTH,
                agent,
                self.centralized_critic,
                self.messages
            ) for agent in range(5)
        }

        # Load agent and critic networks if required
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
            self.centralized_critic.load_network(self.checkpoint_critic)
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()
            self.centralized_critic.load_last_epoch(self.last_checkpoint_file_critic)
    
    # Run the MAPPO training loop
    def run(self):

        self.initialize_environment()
        reward_normalizer = RewardNormalizer()  # Create a reward normalizer to stabilize training

        for i in range(self.max_eps):
            # Reset the environment at the start of each episode
            observations, _ = self.env.reset()
            r = []

            for j in range(self.EPISODE_LENGTH):
                self.count += 1

                # Prepare global state for the centralized critic
                observations_list = self.concatenate_observations(observations, self.agents)
                state_value = self.centralized_critic.get_state_value(observations_list)

                # Collect actions and messages from all agents
                actions_messages = {
                    agent_name: agent.get_action(
                        observations[agent_name],
                        state_value
                    )
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }
                actions = {agent_name: action for agent_name, (action, _) in actions_messages.items()}
                messages = {agent_name: message for agent_name, (_, message) in actions_messages.items()}

                # Execute actions in the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Save episode data to agent memory
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]
                    agent.memory.save_end_episode(reward_normalizer.normalize(reward[agent_name]), done, observations_list)

                # Check for termination of all agents
                done = {agent: termination.get(agent, False) or truncation.get(agent, False) for agent in self.env.agents}
                if all(done.values()):
                    break

                # Collect rewards
                r.append(mean(reward.values()))

            # Update reward statistics
            self.partial_rewards += sum(r)
            self.total_rewards.append(sum(r))
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {mean(self.total_rewards)}")

            # Perform policy update every `rollout` episodes
            if (i + 1) % self.rollout == 0:
                avg_rwd = self.partial_rewards / self.rollout
                self.average_rewards.append(avg_rwd)
                print(f"Average reward obtained before update: {avg_rwd}")

                # Save best performing agent networks
                if avg_rwd > self.best_reward:
                    self.best_reward = avg_rwd
                    for agent_name, agent in self.agents.items():
                        save_agent_network(agent.actor, agent.actor.actor_optimizer, agent.checkpoint_file_actor)
                    save_agent_network(self.centralized_critic, self.centralized_critic.critic_optimizer, self.checkpoint_critic)

                self.partial_rewards = 0

            # Save episodic data and update policies
            for agent_name, agent in self.agents.items():
                agent.memory.save_episode()
                if (i + 1) % self.rollout == 0:
                    print(f"Policy update for {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count)

        # Save final training data and networks
        save_agent_data_ppo(self.agents)
        for agent_name, agent in self.agents.items():
            save_agent_network(agent.actor, agent.actor.actor_optimizer, agent.last_checkpoint_file_actor)
        save_agent_network(self.centralized_critic, self.centralized_critic.critic_optimizer, self.last_checkpoint_file_critic)
        save_statistics(self.total_rewards, self.average_rewards)
