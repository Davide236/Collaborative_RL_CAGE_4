from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.QMIX.qmix import QMix
from CybORG.Agents.QMIX.buffer import ReplayBuffer
from statistics import mean, stdev
import csv
import matplotlib.pyplot as plt


class QMIXTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 1000
    LOAD_NETWORKS = False
    LOAD_BEST = False
    ROLLOUT = 5

    def __init__(self):
        self.env = None
        self.agents = None
        self.memory = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.training_steps = 0

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
            # Add to partial rewards
            self.total_rewards.append(sum(r))
            self.memory.append_episodic()
            if self.memory.ready():
                print("Training...")
                sample = self.memory.sample(self.ROLLOUT)
                self.training_steps += 1
                self.agents.train(sample, self.training_steps)
        self.save_statistics()

    def save_statistics(self):
        rewards_mean = mean(self.total_rewards)
        rewards_stdev = stdev(self.total_rewards)
        total_rewards_transposed = [[elem] for elem in self.average_rewards]
        with open('qmix_reward_history.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Rewards'])  # Write header
            writer.writerows(total_rewards_transposed)
        plt.plot(self.total_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')
        plt.grid(True)
        plt.show()
        print(f"Average reward: {rewards_mean}, standard deviation of {rewards_stdev}")
