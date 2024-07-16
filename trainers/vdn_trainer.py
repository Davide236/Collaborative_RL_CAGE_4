from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.VDN.vdn import VDN
from statistics import mean, stdev
from utils import save_statistics, save_agent_data_mixer, save_agent_network


class VDNTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 1500
    ROLLOUT = 10

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
        n_agents = 3
        actor_dims = []
        agents_actions = []
        # Dimension of observation_space (for each agent)
        for agent in range(n_agents):
            actor_dims.append(env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(env.get_action_space(f'blue_agent_{agent}')['actions']))
        agents = VDN(
            n_agents=n_agents,
            n_actions=agents_actions,
            actor_dims=actor_dims
        )
        return agents

    @staticmethod
    def transform_observations(obs):
        observations = []
        for i in range(3):
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
        self.agents = self.setup_agents(env)
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            self.agents.load_network()
        if self.load_last_network:
            self.agents.load_last_epoch()

    def run(self):
        self.initialize_environment()
        for eps in range(self.MAX_EPS):
            self.agents.update_epsilon(eps, self.MAX_EPS)
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            self.agents.init_hidden_state()
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                acts = self.agents.get_actions(self.transform_observations(observations))
                actions = {
                    f'blue_agent_{i}': int(acts[i])
                    for i in range(3)
                }
                # Perform action on the environment
                new_observations, reward, termination, truncation, _ = self.env.step(actions)
                # Append the rewards and termination for each agent
                done = []
                for i in range(3):
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                # This terminates if all agents have 'termination=true'
                self.agents.save_memory(obs1, acts, reward2, obs2, done)
                observations = new_observations
                if all(done):
                    break
                r.append(mean(reward.values()))  # Add rewards
            self.partial_rewards += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (eps + 1)}")
            if sum(r) > self.best_reward:
                self.best_reward = sum(r)
                save_agent_network(self.agents.q_network, self.agents.optimizer, self.agents.save_best_path)
            # Add to partial rewards
            self.total_rewards.append(sum(r))
            if self.agents.memory.size() > 10000:
                self.agents.train()
            if eps % self.agents.update_interval == 0:
                self.agents.copy_network()
        save_agent_network(self.agents.q_network, self.agents.optimizer, self.agents.save_last_path)
        save_statistics(self.total_rewards, self.total_rewards)
        save_agent_data_mixer(self.agents)