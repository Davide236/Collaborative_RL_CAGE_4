from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.R_MAPPO.r_mappo import PPO
from CybORG.Agents.R_MAPPO.critic_network import CriticNetwork
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
import numpy as np
import torch
import yaml
import os
from utils import save_statistics, save_agent_data_ppo, save_agent_network



class RecurrentMAPPOTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 4000
    ROLLOUT = 10

    def __init__(self, args):
        self.env = None
        self.agents = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.best_reward = -8000
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.best_critic = None
        self.last_critic = None
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.centralized_critic = None
        self.critic_optimizer = None

    def concatenate_observations(self, observations):
        observation_list = []
        for agent_name in self.agents.keys():
            observation_list.extend(observations[agent_name])
        normalized_state = (observation_list - np.mean(observation_list)) / (np.std(observation_list) + 1e-8)
        state = torch.FloatTensor(normalized_state.reshape(1, -1))
        return state

    def initialize_critic(self, env):
        config_file_path = os.path.join(os.path.dirname(__file__), '../CybORG/Agents/MAPPO/hyperparameters.yaml')
        with open(config_file_path, 'r') as file:
            params = yaml.safe_load(file)
        lr = float(params.get('lr', 2.5e-4))
        eps = float(params.get('eps', 1e-5))
        fc = int(params.get('fc', 256))
        centralized_critic = CriticNetwork(env.observation_space('blue_agent_4').shape[0], env.observation_space('blue_agent_0').shape[0], 3, fc)
        critic_optimizer = torch.optim.Adam(centralized_critic.parameters(), lr=lr, eps=eps)
        message_type = params.get('message_type', 'simple')
        return centralized_critic, critic_optimizer, message_type


    def init_checkpoint(self):
        checkpoint_file_critic = os.path.join('saved_networks', f'r_critic_mappoppo_central')
        last_checkpoint_file_critic = os.path.join('last_networks', f'r_critic_mappo_central')
        return checkpoint_file_critic, last_checkpoint_file_critic

    def setup_agents(self, env):
        agents = {f"blue_agent_{agent}": PPO(env.observation_space(f'blue_agent_{agent}').shape[0], len(env.get_action_space(f'blue_agent_{agent}')['actions']), self.MAX_EPS * self.EPISODE_LENGTH, agent, self.centralized_critic, self.critic_optimizer, self.messages) for agent in range(3)}
        return agents

    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        self.env = BlueFlatWrapper(env=cyborg)
        self.env.reset()
        self.best_critic, self.last_critic = self.init_checkpoint()
        self.centralized_critic, self.critic_optimizer, self.message_type = self.initialize_critic(self.env)
        self.checkpoint_critic = os.path.join(f'saved_networks\mappo\{self.message_type}', f'critic_ppo_central')
        self.last_checkpoint_file_critic = os.path.join(f'last_networks\mappo\{self.message_type}', f'critic_ppo_central')
        self.agents = self.setup_agents(self.env)
        print(f'Using agents {self.agents}')
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

    def run(self):
        self.initialize_environment()
        for i in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            for agent_name, agent in self.agents.items():
                agent.set_initial_state(1)
            self.centralized_critic.get_init_state(1)
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                observations_list = self.concatenate_observations(observations)
                state_value = self.centralized_critic(observations_list)
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
                # If all agents are done (truncation) then end the episode
                if all(done.values()):
                    break
                r.append(mean(reward.values()))  # Add rewards
            # Add to partial rewards
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
                        save_agent_network(agent.actor, agent.actor_optimizer, agent.checkpoint_file_actor)
                    save_agent_network(self.centralized_critic, self.critic_optimizer,self.checkpoint_critic)
                self.partial_rewards = 0
            # Save rewards, state values and termination flags (divided per episodes)
            for agent_name, agent in self.agents.items():
                agent.memory.append_episodic()
                # Every 5 episodes perform a policy update
                if (i + 1) % self.ROLLOUT == 0:
                    print(f"Policy update for  {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count)
        save_agent_data_ppo(self.agents)
        for agent_name, agent in self.agents.items():
            save_agent_network(agent.actor, agent.actor_optimizer, agent.last_checkpoint_file_actor)
        save_agent_network(self.centralized_critic, self.critic_optimizer,self.last_checkpoint_file_critic)
        save_statistics(self.total_rewards, self.average_rewards)

