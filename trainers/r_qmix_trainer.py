from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.R_QMIX.qmix import QMix
from CybORG.Agents.R_QMIX.buffer import ReplayBuffer
from statistics import mean
import os
from utils import save_statistics, save_agent_data_mixer, save_agent_network, RewardNormalizer

# Trainer Class for the Recurrent version of the QMIX algorithm
class R_QMIXTrainer:
    # Standard length of CybORG episode
    EPISODE_LENGTH = 500  # Define the number of steps for each episode

    def __init__(self, args):
        # Initialize the trainer class with necessary parameters and states
        self.env = None  # The environment for training
        self.agents = None  # List of agents
        self.memory = None  # Replay buffer for storing experiences
        self.total_rewards = []  # List of total rewards from all episodes
        self.partial_rewards = 0  # Cumulative reward for the current batch of episodes
        self.average_rewards = []  # List of average rewards per batch
        self.count = 0  # Counter for total training steps
        self.training_steps = 0  # Counter for training steps
        self.best_reward = -7000  # Best reward achieved, initialized to a low value
        self.load_last_network = args.Load_last  # Flag to load the last saved network
        self.load_best_network = args.Load_best  # Flag to load the best network
        self.messages = args.Messages  # Flag to enable or disable messages in the environment
        self.rollout = args.Rollout  # Number of episodes per rollout
        self.max_eps = args.Episodes  # Maximum number of episodes to train

    def setup_agents(self, env):
        # Setup agents for training based on the environment
        n_agents = 5  # Define the number of agents
        actor_dims = []  # List to store the dimension of the observation space for each agent
        agents_actions = []  # List to store the number of actions available for each agent

        # Iterate over agents and setup their observation space and action space
        for agent in range(n_agents):
            actor_dims.append(env.observation_space(f'blue_agent_{agent}').shape[0])  # Get the observation space dimensions
            agents_actions.append(len(env.get_action_space(f'blue_agent_{agent}')['actions']))  # Get number of actions

        # Define the critic dimensions as the sum of all agent observation dimensions
        critic_dims = sum(actor_dims)
        
        # Initialize the QMIX agents with the provided dimensions
        agents = QMix(
            n_agents=n_agents,
            n_actions=agents_actions,
            obs_space=actor_dims,
            state_space=critic_dims,
            episode_length=self.EPISODE_LENGTH - 1,  # Set episode length for each agent
            total_episodes=self.EPISODE_LENGTH,  # Set total episode length
            messages=self.messages  # Whether to use messages in the environment
        )
        
        # Initialize the replay buffer with a maximum capacity
        memory = ReplayBuffer(
            1_000_000,  # Max size of the buffer
            actor_dims,  # Dimensions of the agent's observation space
            batch_size=self.rollout,  # Batch size for sampling experience
            episode_length=self.EPISODE_LENGTH - 1  # Episode length
        )

        return agents, memory  # Return initialized agents and memory buffer

    def transform_observations(self, obs):
        # Helper function to transform and concatenate observations for each agent
        observations = []
        for i in range(5):  # For 5 agents
            observations.append(obs[f'blue_agent_{i}'])  # Extract the observation of each agent
        return observations  # Return the list of transformed observations

    def initialize_environment(self):
        # Initialize the environment and agents
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,  # Define the blue agent class (used for the blue team)
            green_agent_class=EnterpriseGreenAgent,  # Define the green agent class (used for the green team)
            red_agent_class=FiniteStateRedAgent,  # Define the red agent class (used for the red team)
            steps=self.EPISODE_LENGTH  # Define the number of steps in each episode
        )
        
        # Initialize the CybORG simulator with the given scenario generator and a random seed
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Seed ensures repeatable results
        
        # Wrap the environment in BlueFlatWrapper (pads observation space for QMIX)
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        env.reset()  # Reset the environment to its initial state
        
        # Store the environment instance
        self.env = env
        
        # Setup agents and memory using the environment
        self.agents, self.memory = self.setup_agents(env)
        
        print(f'Using agents {self.agents}')  # Output information about the agents
        
        # Load pre-trained networks if specified
        if self.load_best_network:
            self.agents.load_network()  # Load the best saved network
        if self.load_last_network:
            self.agents.load_last_epoch()  # Load the last saved network

    def run(self):
        # Main training loop to run the agent interactions with the environment
        self.initialize_environment()  # Initialize environment and agents
        reward_normalizer = RewardNormalizer()  # Initialize the reward normalizer to stabilize rewards
        
        # Loop over episodes
        for eps in range(self.max_eps):  # Iterate through each episode
            self.agents.reset_hidden_layer()  # Reset hidden layers for recurrent networks (if used)
            # Reset environment for each episode and get initial observations
            observations, _ = self.env.reset()
            r = []  # List to store the rewards for each timestep in the episode

            # Iterate through timesteps in the episode
            for j in range(self.EPISODE_LENGTH):  # For each timestep in the episode
                self.count += 1  # Increment the global step counter

                # Action selection for all agents
                acts, msg = self.agents.choose_actions(self.transform_observations(observations))
                actions = {
                    f'blue_agent_{i}': acts[i]  # Map actions to each agent
                    for i in range(5)  # Iterate over all agents
                }

                # If messages are enabled, pass messages along with actions, otherwise just pass actions
                if self.messages:
                    messages = {
                        f'blue_agent_{i}': msg[i]  # Map messages to each agent
                        for i in range(5)
                    }
                    new_observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    new_observations, reward, termination, truncation, _ = self.env.step(actions)

                # Check if each agent has terminated or been truncated
                done = []
                for i in range(5):
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                
                # Transform observations for current and next timestep, and rewards
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                
                # Store the episode experience in the replay buffer
                self.memory.store_episodic(obs1, acts, reward2, obs2, done, step=j)
                observations = new_observations  # Update observations for the next timestep

                # If all agents are done, terminate the episode
                if all(done):
                    break

                r.append(mean(reward.values()))  # Append the mean of rewards for this timestep

            # Accumulate total rewards for the episode
            self.partial_rewards += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (eps + 1)}")

            # Save networks if the current episode exceeds the best reward
            if sum(r) > self.best_reward:
                self.best_reward = sum(r)
                # Save the networks for each agent and the mixing network (QMIX network)
                for number, network in enumerate(self.agents.agent_networks):
                    save_path = os.path.join(f'saved_networks/r_qmix/{self.agents.message_type}', f'qmix_{number}')
                    save_agent_network(network, self.agents.agent_optimizers[number], save_path)
                save_path = os.path.join(f'saved_networks/r_qmix/{self.agents.message_type}', f'mixer')
                save_agent_network(self.agents.qmix_net_eval, self.agents.mixing_optimizer, save_path)

            # Add total rewards of this episode to the list
            self.total_rewards.append(sum(r))
            self.memory.append_episodic()  # Append the episodic memory to the replay buffer

            # Train the agent if the replay buffer has enough samples
            if self.memory.ready():
                print("Training...")
                sample, indices, _ = self.memory.sample(self.rollout)  # Sample a batch from memory
                self.training_steps += 1  # Increment training steps
                td_errors = self.agents.train(sample, self.training_steps)  # Train the agents
                self.memory.set_priorities(indices, td_errors)  # Update priorities for the samples

        # After training, save the final networks and statistics
        for number, network in enumerate(self.agents.agent_networks):
            save_path = os.path.join(f'last_networks\qmix\{self.agents.message_type}', f'qmix_{number}')
            save_agent_network(network, self.agents.agent_optimizers[number], save_path)
        save_path = os.path.join(f'last_networks\qmix\{self.agents.message_type}', f'mixer')
        save_agent_network(self.agents.qmix_net_eval, self.agents.mixing_optimizer, save_path)
        save_statistics(self.total_rewards, self.total_rewards)  # Save the training statistics
        save_agent_data_mixer(self.agents)  # Save agent data and mixer information
