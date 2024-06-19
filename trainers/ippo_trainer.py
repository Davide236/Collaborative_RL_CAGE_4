from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.IPPO.ippo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev

import csv
import matplotlib.pyplot as plt

class PPOTrainer:
    EPISODE_LENGTH = 500
    MAX_EPS = 4000
    LOAD_NETWORKS = False
    LOAD_BEST = False
    MESSAGES = True
    ROLLOUT = 10

    def __init__(self, args):
        self.agents = {}
        self.total_rewards = []
        self.average_rewards = []
        self.partial_rewards = 0
        self.best_reward = -2000
        self.count = 0  # Keep track of total episodes
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages

    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                         green_agent_class=EnterpriseGreenAgent,
                                         red_agent_class=FiniteStateRedAgent,
                                         steps=self.EPISODE_LENGTH)
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        env = BlueFlatWrapper(env=cyborg)
        env.reset()
        self.env = env
        self.agents = {f"blue_agent_{agent}": PPO(env.observation_space(f'blue_agent_{agent}').shape[0],
                                                  len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                                                  self.MAX_EPS*self.EPISODE_LENGTH, agent, self.messages) 
                       for agent in range(5)}
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
                if self.MESSAGES:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Append the rewards and termination for each agent
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]
                    agent.memory.save_end_episode(reward[agent_name], done)
                # This terminates if all agent have 'termination=true'
                done = {
                    agent: termination.get(agent, False) or truncation.get(agent, False)
                    for agent in self.env.agents
                }
                # If all agents are done (truncation) then end the episode
                if all(done.values()):
                    break
                r.append(mean(reward.values()))  # Add rewards  
            print(f"Final reward of the episode: {sum(r)}, length {self.count}")
            # Add to partial rewards  
            self.partial_rewards += sum(r)
            self.total_rewards.append(sum(r))
            # Print average reward before rollout
            if (i+1) % self.ROLLOUT == 0:
                avg_rwd = self.partial_rewards/self.ROLLOUT
                avg = sum(self.average_rewards) + self.partial_rewards
                self.average_rewards.append(avg_rwd)
                print(f"Average reward obtained before update: {avg_rwd}, avg: {avg/(i+1)}")
                # If the average reward is better than the best reward then save agents
                if avg_rwd > self.best_reward:
                    self.best_reward = avg_rwd
                    for agent_name, agent in self.agents.items():
                        agent.save_network()
                self.partial_rewards = 0  
            # Save rewards, state values and termination flags (divided per episodes)    
            for agent_name, agent in self.agents.items():
                agent.memory.save_episode()
                # Every 5 episodes perform a policy update
                if (i+1) % self.ROLLOUT == 0:
                    print(f"Policy update for  {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count) 
        self.save_statistics()

    def save_statistics(self):
        # Save loss data
        for agent_name, agent in self.agents.items():
            agent.save_statistics_csv() 
            agent.save_last_epoch() 
        # Graph of average rewards and print output results 
        rewards_mean = mean(self.total_rewards)
        rewards_stdev = stdev(self.total_rewards)
        total_rewards_transposed = [[elem] for elem in self.average_rewards]
        with open('reward_history.csv', mode='w', newline='') as file:
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
