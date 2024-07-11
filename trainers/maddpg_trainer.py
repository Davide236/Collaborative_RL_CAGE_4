from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
from CybORG.Agents.MADDPG.maddpg import MADDPG
from CybORG.Agents.MADDPG.replay_buffer import MultiAgentReplayBuffer

import csv
import matplotlib.pyplot as plt
from utils import save_statistics, save_agent_data_maddpg, save_agent_network


class MADDPGTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 1000
    LOAD_NETWORKS = False
    LOAD_BEST = False
    ROLLOUT = 5

    def __init__(self, args):
        self.env = None
        self.agents = None
        self.memory = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.best_reward = -7000
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages

    def setup_agents(self):
        n_agents = 7
        actor_dims = []
        agents_actions = []
        # Dimension of observation_space (for each agent)
        for agent in range(n_agents):
            actor_dims.append(self.env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(self.env.get_action_space(f'blue_agent_{agent}')['actions']))
        critic_dims = sum(actor_dims)
        agents = MADDPG(actor_dims, critic_dims, n_agents, agents_actions)
        memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                            agents_actions, n_agents, batch_size=3000)
        return agents, memory

    @staticmethod
    def transform_observations(obs):
        observations = []
        for i in range(7):
            observations.append(obs[f'blue_agent_{i}'])
        return observations

    @staticmethod
    def concatenate_observations(observations):
        observations_list = []
        for i in range(7):
            agent_name = f'blue_agent_{i}'
            observations_list.extend(observations[agent_name])
        return observations_list

    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                         green_agent_class=EnterpriseGreenAgent,
                                         red_agent_class=FiniteStateRedAgent,
                                         steps=self.EPISODE_LENGTH)
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        env = BlueFlatWrapper(env=cyborg)
        env.reset()
        self.env = env
        self.agents, self.memory = self.setup_agents()
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            for agent in self.agents.agents:
                agent.load_network()
        if self.load_last_network:
            for agent in self.agents.agents:
                agent.load_last_epoch()

    def run(self):
        self.initialize_environment()
        for eps in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                acts = self.agents.choose_actions(observations, evaluate=False)
                actions = {
                    f'blue_agent_{i}': acts[i]
                    for i in range(7)
                }
                old_central_observations = self.concatenate_observations(observations)
                # Perform action on the environment
                new_observations, reward, termination, truncation, _ = self.env.step(actions)
                new_central_observations = self.concatenate_observations(new_observations)
                # Append the rewards and termination for each agent
                done = []
                for i in range(7):
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                # This terminates if all agents have 'termination=true'
                self.memory.store_transition(obs1, old_central_observations, acts, reward2, obs2, new_central_observations, done)
                observations = new_observations
                if all(done):
                    break
                r.append(mean(reward.values()))  # Add rewards  
            self.partial_rewards += sum(r)
            self.average_rewards.append(sum(r))
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (eps + 1)}")
            if (sum(r) > self.best_reward):
                for agent in self.agents.agents:
                    save_agent_network(agent.actor, agent.actor.optimizer, agent.checkpoint_file_actor)
                    save_agent_network(agent.critic, agent.critic.optimizer, agent.checkpoint_file_critic)
            # Add to partial rewards  
            self.total_rewards.append(sum(r))
            self.agents.learn(self.memory)
        for agent in self.agents.agents:
            save_agent_network(agent.actor, agent.actor.optimizer, agent.last_checkpoint_file_actor)
            save_agent_network(agent.critic, agent.critic.optimizer, agent.last_checkpoint_file_critic)
        save_agent_data_maddpg(self.agents)
        save_statistics(self.total_rewards, self.average_rewards)


