from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.QMIX.qmix import QMix
from CybORG.Agents.QMIX.buffer import ReplayBuffer
from statistics import mean, stdev
import os
import matplotlib.pyplot as plt
from utils import save_statistics, save_agent_data_mixer, save_agent_network, rewards_handler



class QMIXTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 3000
    ROLLOUT = 5

    def __init__(self, args):
        self.env = None
        self.agents = None
        self.memory = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.training_steps = 0
        self.best_reward = -7000
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages

    def setup_agents(self, env):
        n_agents = 5
        actor_dims = []
        agents_actions = []
        # Dimension of observation_space (for each agent)
        for agent in range(n_agents):
            actor_dims.append(env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(env.get_action_space(f'blue_agent_{agent}')['actions']))
        critic_dims = sum(actor_dims)
        agents = QMix(
            n_agents=n_agents,
            n_actions=agents_actions,
            obs_space=actor_dims,
            state_space=critic_dims,
            episode_length=self.EPISODE_LENGTH - 1,
            total_episodes=self.EPISODE_LENGTH
        )
        memory = ReplayBuffer(
            1_000_000,
            actor_dims,
            batch_size=self.ROLLOUT,
            episode_length=self.EPISODE_LENGTH - 1
        )
        return agents, memory

    @staticmethod
    def transform_observations(obs):
        observations = []
        for i in range(5):
            observations.append(obs[f'blue_agent_{i}'])
        return observations

    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        # Padding required for QMIX
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        env.reset()
        self.env = env
        self.agents, self.memory = self.setup_agents(env)
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            self.agents.load_network()
        if self.load_last_network:
            self.agents.load_last_epoch()

    def run(self):
        self.initialize_environment()
        for eps in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                acts = self.agents.choose_actions(self.transform_observations(observations))
                actions = {
                    f'blue_agent_{i}': acts[i]
                    for i in range(5)
                }
                # Perform action on the environment
                new_observations, reward, termination, truncation, _ = self.env.step(actions)
                reward = rewards_handler(reward)
                # Append the rewards and termination for each agent
                done = []
                for i in range(5):
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                # This terminates if all agents have 'termination=true'
                self.memory.store_episodic(obs1, acts, reward2, obs2, done, step=j)
                observations = new_observations
                if all(done):
                    break
                r.append(mean(reward.values()))  # Add rewards
            self.partial_rewards += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (eps + 1)}")
            if sum(r) > self.best_reward:
                self.best_reward = sum(r)
                for number, network in enumerate(self.agents.agent_networks):
                    save_path = os.path.join(f'saved_networks\qmix\{self.agents.message_type}', f'qmix_{number}')
                    save_agent_network(network, self.agents.agent_optimizers[number], save_path)
                save_path = os.path.join(f'saved_networks\qmix\{self.agents.message_type}', f'mixer')
                save_agent_network(self.agents.qmix_net_eval, self.agents.mixing_optimizer, save_path)
            # Add to partial rewards
            self.total_rewards.append(sum(r))
            self.memory.append_episodic()
            if self.memory.ready():
                print("Training...")
                sample = self.memory.sample(self.ROLLOUT)
                self.training_steps += 1
                self.agents.train(sample, self.training_steps)
        for number, network in enumerate(self.agents.agent_networks):
            save_path = os.path.join(f'last_networks\qmix\{self.agents.message_type}', f'qmix_{number}')
            save_agent_network(network, self.agents.agent_optimizers[number], save_path)
        save_path = os.path.join(f'last_networks\qmix\{self.agents.message_type}', f'mixer')
        save_agent_network(self.agents.qmix_net_eval, self.agents.mixing_optimizer, save_path)
        save_statistics(self.total_rewards, self.total_rewards)
        save_agent_data_mixer(self.agents)
