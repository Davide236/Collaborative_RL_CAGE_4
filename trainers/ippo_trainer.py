from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.IPPO.ippo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean

from utils import save_statistics, save_agent_data_ppo, save_agent_network, RewardNormalizer

# Trainer class for the Proximal Policy Optimization (PPO) algorithm
class PPOTrainer:
    # Standard length of a CybORG episode (number of steps per episode)
    EPISODE_LENGTH = 500

    def __init__(self, args):
        # Initialize training settings and configurations
        self.agents = {}  # Dictionary to store PPO agents
        self.total_rewards = []  # List to track cumulative rewards across episodes
        self.average_rewards = []  # List to track average rewards for rollouts
        self.partial_rewards = 0  # Accumulator for rewards in the current rollout
        self.best_reward = -7000  # Best average reward seen so far
        self.count = 0  # Total steps taken across all episodes
        self.load_last_network = args.Load_last  # Flag to load the last saved network
        self.load_best_network = args.Load_best  # Flag to load the best saved network
        self.messages = args.Messages  # Enable or disable message passing between agents
        self.rollout = args.Rollout  # Number of episodes before policy update
        self.max_eps = args.Episodes  # Total number of episodes for training

    def initialize_environment(self):
        # Set up the CybORG environment and PPO agents
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add seed for reproducibility
        env = BlueFlatWrapper(env=cyborg)  # Wrap environment for flattened blue agent observation/action space
        env.reset()  # Reset environment to initial state
        self.env = env

        # Initialize PPO agents for each blue agent (total 5 agents)
        self.agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                self.max_eps * self.EPISODE_LENGTH,  # Total training steps
                agent,
                self.messages  # Use message passing if enabled
            )
            for agent in range(5)
        }
        print(f'Using agents {self.agents}')

        # Load previously saved network weights if specified
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()  # Load the best network checkpoint
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()  # Load the last saved checkpoint

    def run(self):
        self.initialize_environment()
        reward_normalizer = RewardNormalizer()  # Normalizes rewards for stability
        exploration_normalizer = RewardNormalizer(max_value=50)
        for i in range(self.max_eps):
            # Start a new training episode
            observations, _ = self.env.reset()  # Reset environment to the initial state
            r = []  # List to track rewards for this episode
            for j in range(self.EPISODE_LENGTH):
                self.count += 1  # Increment step counter

                # Get actions and messages for each agent
                actions_messages = {
                    agent_name: agent.get_action(observations[agent_name])
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }

                # Separate actions and messages
                actions = {agent_name: action for agent_name, (action, _) in actions_messages.items()}
                messages = {agent_name: message for agent_name, (_, message) in actions_messages.items()}

                # Compute intrinsic rewards for exploration (optional)
                intrinsic_rewards = {
                    agent_name: agent.get_exploration_reward(observations[agent_name])
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }

                # Perform the action step in the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Update agent memory with rewards and intrinsic exploration rewards
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]
                    agent.memory.save_end_episode(
                        reward_normalizer.normalize(reward[agent_name]),
                        exploration_normalizer.normalize(intrinsic_rewards[agent_name]),
                        done
                    )

                # Check if all agents have terminated
                done = {
                    agent: termination.get(agent, False) or truncation.get(agent, False)
                    for agent in self.env.agents
                }
                if all(done.values()):  # Exit loop if all agents are done
                    break

                r.append(mean(reward.values()))  # Collect the mean reward for this step

            # Update rewards and print episode summary
            self.partial_rewards += sum(r)
            self.total_rewards.append(sum(r))
            print(f"Final reward of the episode: {sum(r)}, steps: {self.count} - AVG: {mean(self.total_rewards)}")

            # Perform policy updates at the end of each rollout
            if (i + 1) % self.rollout == 0:
                avg_rwd = self.partial_rewards / self.rollout  # Average reward in the rollout
                avg = sum(self.average_rewards) + self.partial_rewards
                self.average_rewards.append(avg_rwd)
                print(f"Average reward before update: {avg_rwd}, cumulative avg: {avg / (i + 1)}")

                # Save agents if the average reward exceeds the best reward
                if avg_rwd > self.best_reward:
                    self.best_reward = avg_rwd
                    for agent_name, agent in self.agents.items():
                        save_agent_network(agent.policy.actor, agent.policy.actor_optimizer, agent.checkpoint_file_actor)
                        save_agent_network(agent.policy.critic, agent.policy.critic_optimizer, agent.checkpoint_file_critic)

                self.partial_rewards = 0  # Reset partial rewards for the next rollout

            # Save episode data to memory and perform learning
            for agent_name, agent in self.agents.items():
                agent.memory.save_episode()
                if (i + 1) % self.rollout == 0:
                    print(f"Policy update for {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count)

        # Save final results and network states
        save_agent_data_ppo(self.agents)
        for agent_name, agent in self.agents.items():
            save_agent_network(agent.policy.actor, agent.policy.actor_optimizer, agent.last_checkpoint_file_actor)
            save_agent_network(agent.policy.critic, agent.policy.critic_optimizer, agent.last_checkpoint_file_critic)
        save_statistics(self.total_rewards, self.average_rewards)
