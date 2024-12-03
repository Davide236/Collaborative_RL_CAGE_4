from CybORG.Agents.MADDPG.agent import Agent
import torch as T
import torch.nn.functional as F
from CybORG.Agents.MADDPG.gradient import GST

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, messages):
        """
        Args:
            actor_dims (list): A list of dimensions for the actor network of each agent.
            critic_dims (int): The dimensions for the critic network input (same for all agents).
            n_agents (int): The number of agents in the environment.
            n_actions (list): A list containing the number of actions for each agent.
            messages (bool): Whether the agent should use messages for communication.

        Returns:
            None

        Explanation:
            Initializes the MADDPG class with multiple agents, their respective networks, and a gradient estimator.
            The agents are then initialized and stored in the `agents` list. A gradient estimator is used for action
            selection, and the action space of each agent is considered while creating the agents.
        """
        self.agents = []  # List to store agent instances
        self.n_agents = n_agents  # Total number of agents
        self.n_action = n_actions  # Action space for each agent
        
        # Define the Gumbel softmax temperature and gradient estimator
        gumbel_temp = 1
        gst_gap = 1
        gradient_estimator = GST(gumbel_temp, gst_gap)
        
        # Initialize each agent and store them
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions[agent_idx],
                                      n_actions, agent_idx, n_agents, gradient_estimator, messages))
    
    def choose_actions(self, raw_obs):
        """
        Args:
            raw_obs (dict): A dictionary where each key is an agent's identifier (e.g., 'blue_agent_0')
                            and the value is the observation of that agent.

        Returns:
            (list): A list of chosen actions by all agents.
            (list): A list of messages prepared by the message handler for each agent.

        Explanation:
            This function retrieves the actions for all agents based on their respective observations.
            The actions are determined by the `choose_action` method of each agent, and messages are 
            collected if the agent uses messaging. The actions and messages are returned for further use.
        """
        actions = []  # List to store the actions of all agents
        messages = []  # List to store the messages from all agents
        
        for agent_idx, agent in enumerate(self.agents):
            # Get the action and message for each agent based on its observation
            action, message = agent.choose_action(raw_obs[f'blue_agent_{agent_idx}'])
            actions.append(action)  # Store the chosen action
            messages.append(message)  # Store the message prepared by the agent
        
        return actions, messages  # Return actions and messages for all agents
    
    def learn(self, sample):
        """
        Args:
            sample (dict): A dictionary containing the batch of experience for training.
                - 'central_obs': The central observations for all agents.
                - 'central_obs_next': The next central observations for all agents.
                - 'obs_next': The next observations for each agent.
                - 'actions': The actions taken by the agents.
                - 'rewards': The rewards received by the agents.
                - 'dones': Whether the episode has ended for each agent.

        Returns:
            (list): A list of average losses for each agent over the learning step.

        Explanation:
            This function handles the learning process for all agents by computing the target actions using the
            target networks. It then computes the loss for both the actor and critic networks for each agent, 
            performs backpropagation, and updates the networks. Finally, the function returns the average loss
            for each agent.
        """
        print("Learning...")  # Print learning progress
        central_obs = sample['central_obs'].transpose(0, 1)  # Central observations
        new_central_obs = sample['central_obs_next'].transpose(0, 1)  # Next central observations

        # Get target actions for each agent based on next observations
        target_actions = [
            self.agents[i].target_actions(sample['obs_next'][i])
            for i in range(self.n_agents)
        ]

        # One-hot encoding of the target actions for each agent
        target_actions_one_hot = [
            F.one_hot(target_actions[i], num_classes=self.agents[i].n_actions)
            for i in range(self.n_agents)
        ]

        # One-hot encoding of the sampled actions for each agent
        sample_actions_one_hot = [
            F.one_hot(sample['actions'][i].to(T.int64), num_classes=self.agents[i].n_actions)
            for i in range(self.n_agents)
        ]
        
        rewards = sample['rewards']  # Rewards from the environment
        dones = sample['dones']  # Whether the episode is done for each agent
        total_loss = []  # List to store the total loss for each agent
        
        # Perform learning for each agent
        for i, agent in enumerate(self.agents):
            # Learn critic by calculating the critic's loss
            actor_loss = agent.learn_critic(
                obs=central_obs,
                new_obs=new_central_obs,
                target_actions=target_actions_one_hot,
                sample_actions=sample_actions_one_hot,
                rewards=rewards[i].unsqueeze(dim=1),
                dones=dones[i].unsqueeze(dim=1)
            )
            # Learn actor by calculating the actor's loss
            critic_loss = agent.learn_actor(
                obs=central_obs,
                agent_obs=sample['obs'][i],
                sampled_actions=sample_actions_one_hot
            )
            # Append the total loss (actor + critic loss)
            total_loss.append(actor_loss + critic_loss)
        
        # Update the parameters of all agents' networks
        for agent in self.agents:
            agent.update_network_parameters()
        
        # Return the average loss for each agent
        return [sum(total_loss) / self.n_agents for _ in range(self.n_agents)]
