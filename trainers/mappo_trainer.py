from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.MAPPO.mappo import PPO
from CybORG.Agents.MAPPO.critic_network import CriticNetwork
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
import numpy as np
import torch
import os
import yaml
from utils import save_statistics, save_agent_data_ppo, save_agent_network

class MAPPOTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 4000
    ROLLOUT = 10

    def __init__(self, args):
        self.env = None
        self.agents = None
        self.centralized_critic = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.best_reward = -7000
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages

    @staticmethod
    def concatenate_observations(observations, agents):
        #print(f'Concatenating Observations: {observations}')
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
        # Add the mission phase
        observation_list.insert(0,mission_phase)
        # Add the messages list
        observation_list.extend(messages_0)
        observation_list.extend(messages_1_4)
        # Normalize the array
        normalized_state = (observation_list - np.mean(observation_list)) / (np.std(observation_list) + 1e-8)
        state = torch.FloatTensor(normalized_state.reshape(1, -1))
        return state

    @staticmethod
    def initialize_critic(env):
        config_file_path = os.path.join(os.path.dirname(__file__), '../CybORG/Agents/MAPPO/hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        lr = float(params.get('lr', 2.5e-4))
        eps = float(params.get('eps', 1e-5))
        fc = int(params.get('fc', 256))
        global_state = params.get('global_state', 'standard')
        if global_state == 'standard':
            state_dim = env.observation_space('blue_agent_4').shape[0] + (env.observation_space('blue_agent_0').shape[0])*(5-1)
        else:
            state_dim = 454 # TODO: Change this
        centralized_critic = CriticNetwork(
            state_dim, lr, eps, fc
        )
        message_type = params.get('message_type', 'simple')
        return centralized_critic, message_type

    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        env = BlueFlatWrapper(env=cyborg)
        env.reset()
        self.env = env
        self.centralized_critic, self.message_type = self.initialize_critic(env)
        self.checkpoint_critic = os.path.join(f'saved_networks/mappo/{self.message_type}', f'critic_ppo_central')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks/mappo/{self.message_type}', f'critic_ppo_central')
        self.agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                self.MAX_EPS * self.EPISODE_LENGTH,
                agent,
                self.centralized_critic,
                self.messages
            ) for agent in range(5)
        }
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
            self.centralized_critic.load_network(self.checkpoint_critic)
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()
            self.centralized_critic.load_last_epoch(self.last_checkpoint_file_critic)

    def run(self):
        self.initialize_environment()
        for i in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                observations_list = self.concatenate_observations(observations, self.agents)
                state_value = self.centralized_critic.get_state_value(observations_list)
                # Action selection for all agents
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

                # Perform action on the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Append the rewards and termination for each agent
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]
                    agent.memory.save_end_episode(reward[agent_name], done, observations_list)
                # This terminates if all agents have 'termination=true'
                done = {
                    agent: termination.get(agent, False) or truncation.get(agent, False)
                    for agent in self.env.agents
                }
                if all(done.values()):
                    break
                r.append(mean(reward.values()))  # Add rewards
            self.partial_rewards += sum(r)
            self.total_rewards.append(sum(r))
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {mean(self.total_rewards)}")
            # Print average reward before rollout
            if (i + 1) % self.ROLLOUT == 0:
                avg_rwd = self.partial_rewards / self.ROLLOUT
                self.average_rewards.append(avg_rwd)
                print(f"Average reward obtained before update: {avg_rwd}")
                # If the average reward is better than the best reward then save agents
                if avg_rwd > self.best_reward:
                    self.best_reward = avg_rwd
                    for agent_name, agent in self.agents.items():
                        save_agent_network(agent.actor, agent.actor.actor_optimizer, agent.checkpoint_file_actor)
                    save_agent_network(self.centralized_critic, self.centralized_critic.critic_optimizer,self.checkpoint_critic)
                self.partial_rewards = 0
            # Save rewards, state values, and termination flags (divided per episode)
            for agent_name, agent in self.agents.items():
                agent.memory.save_episode()
                # Every 5 episodes perform a policy update
                if (i + 1) % self.ROLLOUT == 0:
                    print(f"Policy update for {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count)
        save_agent_data_ppo(self.agents)
        for agent_name, agent in self.agents.items():
            save_agent_network(agent.actor, agent.actor.actor_optimizer, agent.last_checkpoint_file_actor)
        save_agent_network(self.centralized_critic, self.centralized_critic.critic_optimizer,self.last_checkpoint_file_critic)
        save_statistics(self.total_rewards, self.average_rewards)

