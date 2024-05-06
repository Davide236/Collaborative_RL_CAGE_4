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
        for agent in self.agents:
            agent.learn(memory, self.agents)

    def learn_second(self, memory):
        # New function here
        pass
    