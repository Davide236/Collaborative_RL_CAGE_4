from CybORG.Agents.MADDPG.agent import Agent
import torch as T
import torch.nn.functional as F
from CybORG.Agents.MADDPG.gradient import GST

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions):
        self.agents = []
        self.n_agents = n_agents
        # TODO: Change here, since actions space is different
        self.n_action = n_actions
        gumbel_temp = 1
        gst_gap = 1
        gradient_estimator = GST(gumbel_temp, gst_gap)
        # Save the agents
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims, n_actions[agent_idx],
                                      n_actions, agent_idx, n_agents, gradient_estimator))
    
    # Get actions from all the agents
    def choose_actions(self, raw_obs, evaluate=False):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[f'blue_agent_{agent_idx}'], evaluate)
            actions.append(action)
        return actions
    
    def learn(self, memory):
        if not memory.ready():
            print("Memory not ready")
            return
        else:
            print("Learning....")
        sample = memory.sample_buffer()
        # Observations samples
        central_obs = T.concat(sample['obs'], axis = 1)
        new_central_obs = T.concat(sample['new_obs'], axis=1)

        # Actions under the target network with the 'next' observations 
        target_actions = [
            self.agents[i].target_actions(sample['new_obs'][i])
            for i in range(self.n_agents)
        ]

        target_actions_one_hot = [
            F.one_hot(target_actions[i], num_classes=self.agents[i].n_actions)
            for i in range(self.n_agents)
        ]

        sample_actions_one_hot = [
            F.one_hot(sample['actions'][i].to(T.int64), num_classes=self.agents[i].n_actions)
            for i in range(self.n_agents)
        ]
        rewards = sample['rewards']
        dones = sample['dones']
        for i, agent in enumerate(self.agents):
            agent.learn_critic(
                obs = central_obs,
                new_obs = new_central_obs,
                target_actions = target_actions_one_hot,
                sample_actions = sample_actions_one_hot,
                rewards = rewards[i].unsqueeze(dim=1),
                dones = dones[i].unsqueeze(dim=1)
            )
            agent.learn_actor(
                obs = central_obs,
                agent_obs = sample['obs'][i],
                sampled_actions = sample_actions_one_hot
            )
        for agent in self.agents:
            agent.update_network_parameters()
    