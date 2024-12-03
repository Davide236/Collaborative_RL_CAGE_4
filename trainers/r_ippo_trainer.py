from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.R_IPPO.ppo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean
from utils import save_statistics, save_agent_data_ppo, save_agent_network, RewardNormalizer


# Trainer Class for the Recurrent version of the IPPO algorithm
class RecurrentIPPOTrainer:
    # Standard length of CybORG episode
    EPISODE_LENGTH = 500

    def __init__(self, args):
        self.env = None  # CybORG simulation environment
        self.agents = None  # Dictionary of agents
        self.total_rewards = []  # Total rewards for all episodes
        self.partial_rewards = 0  # Accumulated rewards for the current batch
        self.reward_before_update = 0  # Rewards accumulated before a policy update
        self.best_reward = -7000  # Best reward achieved during training
        self.load_last_network = args.Load_last  # Flag to load the most recent network
        self.load_best_network = args.Load_best  # Flag to load the best-performing network
        self.messages = args.Messages  # Whether to enable inter-agent messaging
        self.average_rewards = []  # Average rewards across batches
        self.count = 0  # Total step count across all episodes
        self.rollout = args.Rollout  # Number of episodes between policy updates
        self.max_eps = args.Episodes  # Total number of episodes for training

    # Initialize the agents
    def setup_agents(self, env):
        agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],  # Observation space size
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),  # Action space size
                self.max_eps * self.EPISODE_LENGTH,  # Total timesteps
                agent,  # Agent identifier
                self.messages  # Messaging flag
            )
            for agent in range(5)  # Number of agents
        }
        return agents

    # Initialize the CybORG environment and agents
    def initialize_environment(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,  # Blue agent class
            green_agent_class=EnterpriseGreenAgent,  # Green agent class
            red_agent_class=FiniteStateRedAgent,  # Red agent class
            steps=self.EPISODE_LENGTH  # Episode length
        )
        # Create the CybORG environment
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Setting seed for reproducibility
        self.env = BlueFlatWrapper(env=cyborg)  # Wrapper for compatibility with PPO
        self.env.reset()  # Reset the environment

        # Initialize agents and load pre-trained networks if specified
        self.agents = self.setup_agents(self.env)
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()

    # Run the IPPO training loop
    def run(self):
        self.initialize_environment()  # Initialize environment and agents
        reward_normalizer = RewardNormalizer()  # Normalize rewards for stable learning

        # Main training loop
        for i in range(self.max_eps):  # Iterate over episodes
            observations, _ = self.env.reset()  # Reset environment at the start of each episode
            
            # Reset initial state for recurrent networks
            for agent_name, agent in self.agents.items():
                agent.set_initial_state(1)

            r = []  # Accumulator for episode rewards
            for j in range(self.EPISODE_LENGTH):  # Step through the episode
                self.count += 1  # Increment total step count
                
                # Agents select actions and optionally send messages
                actions_messages = {
                    agent_name: agent.get_action(observations[agent_name])
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }
                # Split actions and messages into separate dictionaries
                actions = {agent_name: action for agent_name, (action, _) in actions_messages.items()}
                messages = {agent_name: message for agent_name, (_, message) in actions_messages.items()}

                # Perform action on the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)

                # Store rewards and termination flags for each agent
                for agent_name, agent in self.agents.items():
                    done = termination[agent_name] or truncation[agent_name]
                    agent.memory.save_end_episode(reward_normalizer.normalize(reward[agent_name]), done)

                # Check if all agents are done
                done = {
                    agent: termination.get(agent, False) or truncation.get(agent, False)
                    for agent in self.env.agents
                }
                if all(done.values()):  # End episode if all agents are done
                    break
                
                r.append(mean(reward.values()))  # Add mean reward for this step
            
            # Update total rewards and print episode statistics
            self.partial_rewards += sum(r)
            self.reward_before_update += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (i + 1)}")
            self.total_rewards.append(sum(r))

            # Perform policy update every `rollout` episodes
            if (i + 1) % self.rollout == 0:
                self.average_rewards.append(self.reward_before_update / self.rollout)  # Calculate average reward
                if self.reward_before_update / self.rollout > self.best_reward:  # Save best networks
                    self.best_reward = self.reward_before_update / self.rollout
                    for agent_name, agent in self.agents.items():
                        save_agent_network(agent.actor, agent.actor_optimizer, agent.checkpoint_file_actor)
                        save_agent_network(agent.critic, agent.critic_optimizer, agent.checkpoint_file_critic)
                self.reward_before_update = 0  # Reset reward accumulator

            # Save rewards and train the agents
            for agent_name, agent in self.agents.items():
                agent.memory.append_episodic()  # Finalize episode in memory
                if (i + 1) % self.rollout == 0:  # Train every `rollout` episodes
                    print(f"Policy update for {agent_name}. Total steps: {self.count}")
                    agent.learn(self.count)

        # Save final training data and models
        save_agent_data_ppo(self.agents)  # Save training data
        for agent_name, agent in self.agents.items():  # Save final models
            save_agent_network(agent.actor, agent.actor_optimizer, agent.last_checkpoint_file_actor)
            save_agent_network(agent.critic, agent.critic_optimizer, agent.last_checkpoint_file_critic)
        save_statistics(self.total_rewards, self.average_rewards)  # Save reward statistics
