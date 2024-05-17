import numpy as np
import torch as T

class MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        # Centralized state observations
        # self.state_memory = np.zeros((self.mem_size, critic_dims))
        # self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        # self.reward_memory = np.zeros((self.mem_size, n_agents))
        # self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        #self.action_memory = T.zeros(self.mem_size, self.n_agents)
        self.action_memory = T.zeros(self.n_agents, self.mem_size)
        self.reward_memory = T.zeros(self.n_agents, self.mem_size)
        self.terminal_memory = T.zeros(self.n_agents, self.mem_size)
        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        # self.actor_action_memory = []

        for i in range(self.n_agents):
            #self.actor_state_memory.append(
                    #np.zeros((self.mem_size, self.actor_dims[i])))
            #self.actor_new_state_memory.append(
                    #np.zeros((self.mem_size, self.actor_dims[i])))\
            self.actor_state_memory.append(T.zeros(self.mem_size, self.actor_dims[i]))
            self.actor_new_state_memory.append(T.zeros(self.mem_size, self.actor_dims[i]))
            # self.actor_action_memory.append(
            #         np.zeros((self.mem_size, self.n_actions[i])))

    def store_transition(self, raw_obs, state, actions, reward,
                         raw_obs_, state_, done):
        # TODO: Check this, each obs in save in columns not rows
        index = self.mem_cntr % self.mem_size
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = T.tensor(raw_obs[agent_idx])
            self.actor_new_state_memory[agent_idx][index] = T.tensor(raw_obs_[agent_idx])
        # print(actions)
        # print(self.action_memory)
        # print(self.action_memory[:, index])
        self.action_memory[:,index] = T.tensor(actions)
        #print(self.action_memory[:, index])
        # self.state_memory[index] = state
        # self.new_state_memory[index] = state_
        self.reward_memory[:,index] = T.tensor(reward)
        self.terminal_memory[:,index] = T.tensor(done)
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        return {
            "obs": [self.actor_state_memory[i][batch] for i in range(self.n_agents)],
            "new_obs": [self.actor_new_state_memory[i][batch] for i in range(self.n_agents)],
            "actions": self.action_memory[:, batch],
            "rewards": self.reward_memory[:, batch],
            "dones": self.terminal_memory[:, batch],
        }
        # states = self.state_memory[batch]
        # states_ = self.new_state_memory[batch]
        # rewards = self.reward_memory[batch]
        # terminal = self.terminal_memory[batch]

        # actions = self.action_memory[batch]
        # actor_states = []
        # actor_new_states = []
        # # actions = []
        # for agent_idx in range(self.n_agents):
        #     actor_states.append(self.actor_state_memory[agent_idx][batch])
        #     actor_new_states.append(
        #         self.actor_new_state_memory[agent_idx][batch])
        #     # actions.append(self.actor_action_memory[agent_idx][batch])

        # return actor_states, states, actions, rewards, \
        #     actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False