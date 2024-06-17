import torch
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3
import gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from vdn import VDN



def transform_observations(obs):
        observations = []
        for i in range(3):
            observations.append(obs[f'agent_{i}'])
        return observations

    
def train_VDN_agent(env_name, lr, gamma, batch_size, buffer_limit, log_interval, max_episodes, max_epsilon,
                    min_epsilon,test_episodes, warm_up_steps, update_iter, chunk_size, update_target_interval,
                    recurrent):

    # create env.
    env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    _, _ = env.reset()
    #memory = ReplayBuffer(buffer_limit)
    avg_rewd = []
    actor_dims = []
    n_actions = []
    full_rwd = 0
    for agent in env.agents:
        actor_dims.append(env.observation_space(agent).shape[0])
        n_actions.append(env.action_spaces[agent].n)
    # create networks
    agents = VDN(n_agents=3, n_actions=n_actions, actor_dims=actor_dims)
    
    # For performance monitoring
    n_agents = len(actor_dims)
    EPISODES = 25000
    for episode_i in range(EPISODES):
        agents.update_epsilon(episode_i, EPISODES)
        state, _ = env.reset()
        done = [False for _ in range(n_agents)]
        total_reward = 0
        with torch.no_grad():
            agents.init_hidden_state()
            step_counter = 0
            while True:
                step_counter += 1
                state = transform_observations(state)
                action = agents.get_actions(state)
                actions = {}
                for i in range(3):
                    #actions[agent] = vdnAgents.act(vdnAgents.combine(obs[agent], agent), agent, epsilon)
                    actions[f'agent_{i}'] = int(action[i])
                next_state, rewards, term, trunc, info = env.step(actions)
                terminated = np.array(transform_observations(term))
                truncated = np.array(transform_observations(trunc))
                reward = transform_observations(rewards)
                final_reward = np.sum(reward)
                total_reward += final_reward
                state_x = next_state
                next_state = transform_observations(next_state)
                done = terminated | truncated
                agents.save_memory(state, action, reward, next_state, done)
                #memory.put((state, action, (np.array(reward)).tolist(), next_state, [int(all(done))]))
                state = state_x
                
                if all(done):
                    break
            
                
        if agents.memory.size() > 2000:
            agents.train()
        
        
        full_rwd += total_reward
        avg_rewd.append(full_rwd/(episode_i+1))
        if episode_i % update_target_interval == 0:
            agents.copy_network()

        # if (episode_i + 1) % log_interval == 0:
        #     test_score = test(test_env, test_episodes, q)
        #     print("#{:<10}/{} episodes, test score: {:.1f} n_buffer : {}, eps : {:.2f}"
        #           .format(episode_i, max_episodes, test_score, memory.size(), epsilon))
    
    env.close()
    plt.figure()
    plt.plot(avg_rewd)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.show()
    
def run():
    kwargs = {'env_name': 'dummy',
          'lr': 0.001,
          'batch_size': 32,
          'gamma': 0.99,
          'buffer_limit': 50000, #50000
          'update_target_interval': 20,
          'log_interval': 500,
          'max_episodes': 20000,
          'max_epsilon': 0.9,
          'min_epsilon': 0.25,
          'test_episodes': 5,
          'warm_up_steps': 2000,
          'update_iter': 10,
          'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
          'recurrent': False}
    train_VDN_agent(**kwargs)

if __name__ == '__main__':
    run()
