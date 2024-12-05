from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean
from CybORG.Agents.MADDPG.maddpg import MADDPG
from CybORG.Agents.MADDPG.replay_buffer import ReplayBuffer

from utils import save_statistics, save_agent_data_maddpg, save_agent_network, RewardNormalizer

# Trainer class for the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm
class MADDPGTrainer:
    # Standard length of a CybORG episode (number of steps per episode)
    EPISODE_LENGTH = 500

    def __init__(self, args):
        # Initialize training settings and configurations
        self.env = None  # CybORG environment
        self.agents = None  # MADDPG agents
        self.memory = None  # Replay buffer for experience storage
        self.total_rewards = []  # List to track total rewards across episodes
        self.partial_rewards = 0  # Accumulated rewards for the current rollout
        self.average_rewards = []  # List to track average rewards for rollouts
        self.count = 0  # Total steps taken across all episodes
        self.best_reward = -7000  # Best average reward observed
        self.load_last_network = args.Load_last  # Flag to load the last saved network
        self.load_best_network = args.Load_best  # Flag to load the best saved network
        self.messages = args.Messages  # Enable or disable message passing between agents
        self.max_eps = args.Episodes  # Total number of episodes for training
        self.rollout = args.Rollout # Number of episodes before training updates

    def setup_agents(self):
        # Configure agent observation and action dimensions
        n_agents = 5
        actor_dims = []  # Dimensions of observation spaces for agents
        agents_actions = []  # Number of actions available to each agent

        for agent in range(n_agents):
            actor_dims.append(self.env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(self.env.get_action_space(f'blue_agent_{agent}')['actions']))
        
        critic_dims = sum(actor_dims)  # Combined dimension for the critic (global observation space)
        
        # Initialize MADDPG agents
        agents = MADDPG(actor_dims, critic_dims, n_agents, agents_actions, self.messages)
        
        # Initialize replay buffer for experience storage
        memory = ReplayBuffer(
            capacity=2000,  # Maximum size of the replay buffer
            obs_dims=actor_dims,
            batch_size=self.rollout,  # Batch size for training
            episode_length=self.EPISODE_LENGTH - 1
        )
        
        return agents, memory
    
    def transform_observations(self, obs):
        # Transform the observations into a list format for easier processing
        observations = []
        for i in range(5):
            observations.append(obs[f'blue_agent_{i}'])
        return observations
    
    def concatenate_observations(self, observations):
        # Combine observations of all agents into a single global observation vector
        observations_list = []
        for i in range(5):
            agent_name = f'blue_agent_{i}'
            observations_list.extend(observations[agent_name])
        return observations_list

    def initialize_environment(self):
        # Set up the CybORG environment and MADDPG agents
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add seed for reproducibility
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)  # Wrap environment with flat observation/action space
        env.reset()  # Reset environment to the initial state
        self.env = env

        # Initialize agents and memory buffer
        self.agents, self.memory = self.setup_agents()
        print(f'Using agents {self.agents}')
        
        # Load saved networks if specified
        if self.load_best_network:
            for agent in self.agents.agents:
                agent.load_network()  # Load best network checkpoint
        if self.load_last_network:
            for agent in self.agents.agents:
                agent.load_last_epoch()  # Load last saved checkpoint

    def run(self):
        self.initialize_environment()
        reward_normalizer = RewardNormalizer()  # Create a reward normalizer to stabilize training
        for eps in range(self.max_eps):
            # Start a new training episode
            observations, _ = self.env.reset()
            r = []  # List to track rewards for this episode

            for j in range(self.EPISODE_LENGTH):
                self.count += 1  # Increment step counter

                # Choose actions and optional messages for all agents
                acts, msg = self.agents.choose_actions(observations)
                actions = {f'blue_agent_{i}': acts[i] for i in range(5)}

                # Concatenate observations for centralized critic
                old_central_observations = self.concatenate_observations(observations)

                # Perform an action step in the environment
                if self.messages:
                    messages = {f'blue_agent_{i}': msg[i] for i in range(5)}
                    new_observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    new_observations, reward, termination, truncation, _ = self.env.step(actions)
                
                new_central_observations = self.concatenate_observations(new_observations)
                
                # Determine whether agents are done (terminated or truncated)
                done = [termination[f'blue_agent_{i}'] or truncation[f'blue_agent_{i}'] for i in range(5)]
                
                # Store episodic data in replay buffer
                obs1 = self.transform_observations(observations)
                obs2 = self.transform_observations(new_observations)
                reward2 = self.transform_observations(reward)
                # Normalize rewards
                reward2 = [reward_normalizer.normalize(x) for x in reward2]
                self.memory.store_episodic(
                    obs1, acts, reward2, obs2, done, old_central_observations, new_central_observations, step=j
                )
                observations = new_observations

                # Terminate if all agents are done
                if all(done):
                    break
                
                r.append(mean(reward.values()))  # Collect mean reward for this step
            
            # Update rewards and log episode statistics
            self.partial_rewards += sum(r)
            self.total_rewards.append(sum(r))
            self.average_rewards.append(sum(r))
            print(f"Final reward of the episode: {sum(r)}, steps: {self.count} - AVG: {self.partial_rewards / (eps + 1)}")

            # Save networks if reward exceeds the best observed reward
            if sum(r) > self.best_reward:
                self.best_reward = sum(r)
                for agent in self.agents.agents:
                    save_agent_network(agent.actor, agent.actor.optimizer, agent.checkpoint_file_actor)
                    save_agent_network(agent.critic, agent.critic.optimizer, agent.checkpoint_file_critic)
            
            # Add experience to the buffer
            self.memory.append_episodic()

            # Perform training if the replay buffer has enough experiences
            if self.memory.ready():
                sample, indices, importance = self.memory.sample(self.rollout)
                td_errors = self.agents.learn(sample)
                self.memory.set_priorities(indices, td_errors)
        
        # Save final data and agent networks
        for agent in self.agents.agents:
            save_agent_network(agent.actor, agent.actor.optimizer, agent.last_checkpoint_file_actor)
            save_agent_network(agent.critic, agent.critic.optimizer, agent.last_checkpoint_file_critic)
        save_agent_data_maddpg(self.agents)
        save_statistics(self.total_rewards, self.average_rewards)
