from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean
from CybORG.Agents.MADDPG.maddpg import MADDPG
from CybORG.Agents.MADDPG.replay_buffer import ReplayBuffer

import csv
import matplotlib.pyplot as plt
from utils import save_statistics, save_agent_data_maddpg, save_agent_network

# Trainer Class for the MADDPG algorithm
class MADDPGTrainer:
    # Standard length of CybORG episode
    EPISODE_LENGTH = 500
    LOAD_NETWORKS = False
    LOAD_BEST = False
    ROLLOUT = 10

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
        self.max_eps = args.Episodes

    def setup_agents(self):
        n_agents = 5
        actor_dims = []
        agents_actions = []
        # Dimension of observation_space and action space (for each agent)
        for agent in range(n_agents):
            actor_dims.append(self.env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(self.env.get_action_space(f'blue_agent_{agent}')['actions']))
        critic_dims = sum(actor_dims)
        # Initialize MADDPG agents
        agents = MADDPG(actor_dims, critic_dims, n_agents, agents_actions,self.messages)
        # Initialize replay buffer
        memory = ReplayBuffer(
            2000,
            actor_dims,
            batch_size=self.ROLLOUT,
            episode_length=self.EPISODE_LENGTH - 1
        )
        return agents, memory
    
    # Transform the observations in an array format
    def transform_observations(obs):
        observations = []
        for i in range(5):
            observations.append(obs[f'blue_agent_{i}'])
        return observations
    
    # Function used to concatenate the observation of the different agents into a singular global observation vector
    def concatenate_observations(observations):
        observations_list = []
        for i in range(5):
            agent_name = f'blue_agent_{i}'
            observations_list.extend(observations[agent_name])
        return observations_list

    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                         green_agent_class=EnterpriseGreenAgent,
                                         red_agent_class=FiniteStateRedAgent,
                                         steps=self.EPISODE_LENGTH)
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        env.reset()
        self.env = env
        self.agents, self.memory = self.setup_agents()
        print(f'Using agents {self.agents}')
        # Load previously saved agents' networks
        if self.load_best_network:
            for agent in self.agents.agents:
                agent.load_network()
        if self.load_last_network:
            for agent in self.agents.agents:
                agent.load_last_epoch()

    def run(self):
        self.initialize_environment()
        for eps in range(self.max_eps):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                acts, msg = self.agents.choose_actions(observations)
                actions = {
                    f'blue_agent_{i}': acts[i]
                    for i in range(5)
                }
                old_central_observations = self.concatenate_observations(observations)
                # Perform action on the environment
                if self.messages:
                    messages = {
                        f'blue_agent_{i}': msg[i]
                        for i in range(5)
                    }
                    new_observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    new_observations, reward, termination, truncation, _ = self.env.step(actions)
                new_central_observations = self.concatenate_observations(new_observations)
                # Append the rewards and termination for each agent
                done = []
                for i in range(5):
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                # Store all the episodic data in the buffer
                self.memory.store_episodic(obs1, acts, reward2, obs2, done, old_central_observations, new_central_observations, step=j)
                observations = new_observations
                # This terminates if all agents have 'termination=true'
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
            self.memory.append_episodic()
            # If the memory has enough episodes then perform a training run
            if self.memory.ready():
                sample, indices, importance = self.memory.sample(self.ROLLOUT)
                td_errors = self.agents.learn(sample)
                self.memory.set_priorities(indices, td_errors)
        # Save all the data of the training
        for agent in self.agents.agents:
            save_agent_network(agent.actor, agent.actor.optimizer, agent.last_checkpoint_file_actor)
            save_agent_network(agent.critic, agent.critic.optimizer, agent.last_checkpoint_file_critic)
        save_agent_data_maddpg(self.agents)
        save_statistics(self.total_rewards, self.average_rewards)


