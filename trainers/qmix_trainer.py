from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.QMIX.qmix import QMix
from CybORG.Agents.QMIX.buffer import ReplayBuffer
from statistics import mean
import os
from utils import save_statistics, save_agent_data_mixer, save_agent_network, RewardNormalizer


# Trainer Class for the QMIX algorithm
class QMIXTrainer:
    # Standard length of CybORG episode
    EPISODE_LENGTH = 500

    def __init__(self, args):
        self.env = None  # The CybORG simulation environment
        self.agents = None  # QMIX agents
        self.memory = None  # Replay buffer for storing experiences
        self.total_rewards = []  # Total rewards across all episodes
        self.partial_rewards = 0  # Reward sum for a batch of episodes
        self.average_rewards = []  # Average reward per batch
        self.count = 0  # Total step count across all episodes
        self.training_steps = 0  # Counter for the number of training steps
        self.best_reward = -7000  # Keeps track of the highest reward achieved
        self.load_last_network = args.Load_last  # Flag for loading the last saved network
        self.load_best_network = args.Load_best  # Flag for loading the best-performing network
        self.messages = args.Messages  # Whether to use inter-agent messaging
        self.rollout = args.Rollout  # Number of episodes between training updates
        self.max_eps = args.Episodes  # Total number of episodes for training

    # Initialize the QMIX agents and memory buffer in the environment
    def setup_agents(self, env):
        n_agents = 5  # Number of agents in the environment
        actor_dims = []  # Dimensions of observation space for each agent
        agents_actions = []  # Dimensions of action space for each agent
        
        # Get observation and action space sizes for each agent
        for agent in range(n_agents):
            actor_dims.append(env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(env.get_action_space(f'blue_agent_{agent}')['actions']))
        
        critic_dims = sum(actor_dims)  # Combined observation space size for centralized critic
        
        # Initialize QMIX agents
        agents = QMix(
            n_agents=n_agents,
            n_actions=agents_actions,
            obs_space=actor_dims,
            state_space=critic_dims,
            episode_length=self.EPISODE_LENGTH - 1,
            total_episodes=self.EPISODE_LENGTH,
            messages=self.messages
        )
        
        # Initialize replay buffer
        memory = ReplayBuffer(
            1_000_000,  # Maximum buffer size
            actor_dims,  # Observation dimensions for each agent
            batch_size=self.rollout,  # Number of samples per training step
            episode_length=self.EPISODE_LENGTH - 1  # Length of each episode
        )
        
        return agents, memory

    # Transform observations from dictionary to a list format for agents
    def transform_observations(self, obs):
        observations = []
        for i in range(5):  # For each agent
            observations.append(obs[f'blue_agent_{i}'])
        return observations

    # Initialize the CybORG environment and QMIX agents
    def initialize_environment(self):
        # Scenario generator with pre-defined agent classes and episode length
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        
        # Create the CybORG environment and reset it
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Setting seed for reproducibility
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)  # Wrap with padding for QMIX compatibility
        env.reset()
        self.env = env
        
        # Initialize agents and replay buffer
        self.agents, self.memory = self.setup_agents(env)
        print(f'Using agents {self.agents}')
        
        # Load pre-trained networks if specified
        if self.load_best_network:
            self.agents.load_network()
        if self.load_last_network:
            self.agents.load_last_epoch()

    # Run the QMIX training loop
    def run(self):
        self.initialize_environment()
        reward_normalizer = RewardNormalizer()  # Utility to normalize rewards
        
        # Main training loop
        for eps in range(self.max_eps):  # Iterate over episodes
            observations, _ = self.env.reset()  # Reset environment at start of each episode
            r = []  # Reward accumulator for the episode
            
            for j in range(self.EPISODE_LENGTH):  # Step through each episode
                self.count += 1  # Increment global step counter
                
                # Agents select actions based on observations
                acts, msg = self.agents.choose_actions(self.transform_observations(observations))
                
                # Prepare actions dictionary for the environment
                actions = {
                    f'blue_agent_{i}': acts[i]
                    for i in range(5)
                }
                
                # Include messages if enabled
                if self.messages:
                    messages = {
                        f'blue_agent_{i}': msg[i]
                        for i in range(5)
                    }
                    new_observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    new_observations, reward, termination, truncation, _ = self.env.step(actions)
                
                # Store data in the replay buffer
                done = []
                for i in range(5):  # Check termination for each agent
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                
                # Prepare transition data for replay buffer
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                self.memory.store_episodic(obs1, acts, reward2, obs2, done, step=j)
                
                # Update observations for the next step
                observations = new_observations
                
                # Break loop if all agents are done
                if all(done):
                    break
                
                # Append mean reward for this step
                r.append(mean(reward.values()))
            
            # Update total rewards
            self.partial_rewards += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (eps + 1)}")
            
            # Save networks if a new best reward is achieved
            if sum(r) > self.best_reward:
                self.best_reward = sum(r)
                for number, network in enumerate(self.agents.agent_networks):
                    save_path = os.path.join(f'saved_networks/qmix/{self.agents.message_type}', f'qmix_{number}')
                    save_agent_network(network, self.agents.agent_optimizers[number], save_path)
                save_path = os.path.join(f'saved_networks/qmix/{self.agents.message_type}', f'mixer')
                save_agent_network(self.agents.qmix_net_eval, self.agents.mixing_optimizer, save_path)
            
            # Append total rewards for the episode
            self.total_rewards.append(sum(r))
            self.memory.append_episodic()  # Finalize episode in memory
            
            # Train the QMIX model if enough data is available
            if self.memory.ready():
                print("Training...")
                sample, indices, _ = self.memory.sample(self.rollout)  # Sample a batch
                self.training_steps += 1  # Increment training step counter
                td_errors = self.agents.train(sample, self.training_steps)  # Train agents
                self.memory.set_priorities(indices, td_errors)  # Update buffer priorities
        
        # Save final agent networks and statistics after training
        for number, network in enumerate(self.agents.agent_networks):
            save_path = os.path.join(f'last_networks/qmix/{self.agents.message_type}', f'qmix_{number}')
            save_agent_network(network, self.agents.agent_optimizers[number], save_path)
        save_path = os.path.join(f'last_networks/qmix/{self.agents.message_type}', f'mixer')
        save_agent_network(self.agents.qmix_net_eval, self.agents.mixing_optimizer, save_path)
        save_statistics(self.total_rewards, self.total_rewards)  # Save reward statistics
        save_agent_data_mixer(self.agents)  # Save agent-specific data
