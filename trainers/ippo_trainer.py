from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.IPPO.ippo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev

from utils import save_statistics, save_agent_data_ppo, save_agent_network, RewardNormalizer


class PPOTrainer:
    EPISODE_LENGTH = 500

    def __init__(self, args):
        self.agents = {}
        self.total_rewards = []
        self.average_rewards = []
        self.partial_rewards = 0
        self.best_reward = -8000
        self.count = 0  # Keep track of total episodes
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.rollout = args.Rollout
        self.max_eps = args.Episodes

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
                                                  self.max_eps*self.EPISODE_LENGTH, agent, self.messages) 
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
        reward_normalizer = RewardNormalizer()
        for i in range(self.max_eps):
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

                intrinsic_rewards = {
                    agent_name: agent.get_exploration_reward(
                        observations[agent_name]
                    )
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }
                print(messages)
                # Perform action on the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Append the rewards and termination for each agent
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]
                    # Intrinsic Rew
                    agent.memory.save_end_episode(reward_normalizer.normalize(reward[agent_name]),intrinsic_rewards[agent_name], done)
                # This terminates if all agent have 'termination=true'
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
            if (i+1) % self.rollout == 0:
                avg_rwd = self.partial_rewards/self.rollout
                avg = sum(self.average_rewards) + self.partial_rewards
                self.average_rewards.append(avg_rwd)
                print(f"Average reward obtained before update: {avg_rwd}, avg: {avg/(i+1)}")
                # If the average reward is better than the best reward then save agents
                if avg_rwd > self.best_reward:
                    self.best_reward = avg_rwd
                    for agent_name, agent in self.agents.items():
                        save_agent_network(agent.policy.actor, agent.policy.actor_optimizer, agent.checkpoint_file_actor)
                        save_agent_network(agent.policy.critic, agent.policy.critic_optimizer, agent.checkpoint_file_critic)
                self.partial_rewards = 0  
            # Save rewards, state values and termination flags (divided per episodes)    
            for agent_name, agent in self.agents.items():
                agent.memory.save_episode()
                # Every X episodes perform a policy update
                if (i+1) % self.rollout == 0:
                    print(f"Policy update for  {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count) 
        save_agent_data_ppo(self.agents)
        for agent_name, agent in self.agents.items():
            save_agent_network(agent.policy.actor, agent.policy.actor_optimizer, agent.last_checkpoint_file_actor)
            save_agent_network(agent.policy.critic, agent.policy.critic_optimizer, agent.last_checkpoint_file_critic)
        save_statistics(self.total_rewards, self.average_rewards)

