import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from pettingzoo.mpe import simple_spread_v3
from buffer import ReplayBuffer
from vdn import VDN
import matplotlib.pyplot as plt

def transform_observations(obs):
        observations = []
        for i in range(3):
            observations.append(obs[f'agent_{i}'])
        return observations
       
def run_VDN(env, device):
    name_agent = env.agents
    num_actions = max(space.n for space in env.action_spaces.values())

    input_shape = len(np.array(env.state()).flatten())
    vdnAgents = VDN(name_agent, num_actions, input_shape,episode_length=25, total_episodes=9000)
    actor_dims = []
    n_actions = []
    for agent in env.agents:
        actor_dims.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_spaces[agent].n)

    memory = ReplayBuffer(1_000_000, actor_dims, batch_size=20, episode_length=25)
    MAX_STEPS = 500000
    total_steps = 0
    episode = 0
    training_steps = 0
    full_rwd = 0
    avg_rewd = []
    while total_steps < MAX_STEPS:
        obs, _ = env.reset()
        steps = 0
        terminal = [False] * 3        
        total_reward = 0
        ep_length = 0
        while not any(terminal):
            steps += 1
            actions = {}
            for agent in name_agent:
                #actions[agent] = vdnAgents.act(vdnAgents.combine(obs[agent], agent), agent, epsilon)
                actions[agent] = vdnAgents.choose_actions(vdnAgents.combine(obs, agent))
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            terminated = np.array(transform_observations(terminations))
            truncated = np.array(transform_observations(truncations))
            terminal = terminated | truncated
            rewards = transform_observations(rewards)
            final_reward = np.sum(rewards)
            total_reward += final_reward
            #old_obs = transform_observations(obs)
            old_obs = []
            new_obs = []
            for agent in name_agent:
                old_obs.append(vdnAgents.combine(obs, agent))
                new_obs.append(vdnAgents.combine(next_obs, agent))
            actions = transform_observations(actions)
            obs = next_obs
            #new_obs = transform_observations(next_obs)
            memory.store_episodic(old_obs, actions, rewards, new_obs, terminal, ep_length)
            ep_length +=1
            steps+=1
        memory.append_episodic()
        if memory.ready():
            #print("LEARNING")
            sample = memory.sample(sample_size = 10)
            training_steps += 1
            vdnAgents.train(sample, training_steps)

        episode += 1
        total_steps += steps
        full_rwd += total_reward
        avg_rewd.append(full_rwd/episode)
        print(f'Reward: {total_reward} in {total_steps}, episode {episode} - AVG: {full_rwd/episode}')
    plt.plot(avg_rewd)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.show()

def run():
    device = 'cpu'
    print(f'Running on : {device}')
    parallel_env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    _, _ = parallel_env.reset()
    agents, data = run_VDN(parallel_env, device=device)

if __name__ == '__main__':
    run()
