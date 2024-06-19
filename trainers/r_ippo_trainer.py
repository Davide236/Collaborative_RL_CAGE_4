from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.R_IPPO.ppo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
import csv
import matplotlib.pyplot as plt


class RecurrentIPPOTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 4000
    ROLLOUT = 10

    def __init__(self, args):
        self.env = None
        self.agents = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.reward_before_update = 0
        self.best_reward = -2000
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes

    def setup_agents(self, env):
        agents = {f"blue_agent_{agent}": PPO(
            env.observation_space(f'blue_agent_{agent}').shape[0],
            len(env.get_action_space(f'blue_agent_{agent}')['actions']),
            self.MAX_EPS * self.EPISODE_LENGTH, agent, self.messages)
            for agent in range(5)
        }
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
        self.agents = self.setup_agents(self.env)
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()

    def run(self):
        self.initialize_environment()
        for i in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            for agent_name, agent in self.agents.items():
                agent.set_initial_state(1)
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                actions_messages = {
                    agent_name: agent.get_action(
                        observations[agent_name]
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
                    agent.memory.save_end_episode(reward[agent_name], done)
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
            self.reward_before_update += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (i + 1)}")
            self.total_rewards.append(sum(r))
            if (i + 1) % self.ROLLOUT == 0:
                self.average_rewards.append(self.reward_before_update / self.ROLLOUT)
                if self.reward_before_update / self.ROLLOUT > self.best_reward:
                    self.best_reward = self.reward_before_update / self.ROLLOUT
                    for agent_name, agent in self.agents.items():
                        agent.save_network()
                self.reward_before_update = 0
            # Save rewards, state values and termination flags (divided per episodes)
            for agent_name, agent in self.agents.items():
                agent.memory.append_episodic()
                # Learn at every episode
                if (i + 1) % self.ROLLOUT == 0:
                    print(f"Policy update for  {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count)
        self.save_statistics()

    def save_statistics(self):
        for _, agent in self.agents.items():
            agent.save_statistics_csv()
            agent.save_last_epoch()
        rewards_mean = mean(self.total_rewards)
        rewards_stdev = stdev(self.total_rewards)
        total_rewards_transposed = [[elem] for elem in self.average_rewards]
        with open('r_ippo_reward_history.csv', mode='w', newline='') as file:
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


