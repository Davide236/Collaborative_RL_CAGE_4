import numpy as np
from pettingzoo.mpe import simple_spread_v3
import matplotlib.pyplot as plt
from buffer import ReplayBuffer
from qmix import QMix



def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def transform_observations(obs):
    observations = []
    for i in range(3):
        observations.append(obs[f'agent_{i}'])
    return observations

def normalize(arr):
    return (arr+8)/8

def run():
    ep_length = 50
    parallel_env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=ep_length, continuous_actions=False)
    

    _, _ = parallel_env.reset()
    n_agents = parallel_env.max_num_agents

    actor_dims = []
    n_actions = []
    for agent in parallel_env.agents:
        actor_dims.append(parallel_env.observation_space(agent).shape[0])
        n_actions.append(parallel_env.action_spaces[agent].n)
    # TODO: Check memory
    second_memory = ReplayBuffer(1_000_000, actor_dims, batch_size=20, episode_length=ep_length)
    
    # TODO: This is a bit different in CybORG
    agents = QMix(n_agents=n_agents, n_actions=n_actions,obs_space=actor_dims,state_space=sum(actor_dims), episode_length = ep_length, total_episodes = 9000)
    MAX_STEPS = 500000
    avg_rewd = []
    training_steps = 0
    total_steps = 0
    episode = 0

    # score = evaluate(maddpg_agents, parallel_env, episode, total_steps)
    # eval_scores.append(score)
    # eval_steps.append(total_steps)
    full_rwd = 0
    while total_steps < MAX_STEPS:
        obs, _ = parallel_env.reset()
        terminal = [False] * n_agents
        total_reward = 0
        steps = 0
        ep_length = 0
        while not any(terminal):            
            actions = agents.choose_actions(transform_observations(obs))
            agent_name = ['agent_0', 'agent_1', 'agent_2']
            actions_with_name = {}
            for i in range(3):
                actions_with_name[agent_name[i]] = actions[i]
            obs_next, rewards, terminal, trunc, _ = parallel_env.step(actions_with_name)
            # Save Old state, New State, Rewards, Action and Termination flags
            terminated = np.array(transform_observations(terminal))
            truncated = np.array(transform_observations(trunc))
            terminal = terminated | truncated
            rewards = transform_observations(rewards)
            final_reward = np.sum(normalize(np.array(rewards)))
            total_reward += final_reward
            old_obs = transform_observations(obs)
            obs = obs_next
            new_obs = transform_observations(obs_next)
            second_memory.store_episodic(old_obs, actions, rewards, new_obs, terminal, ep_length)
            ep_length +=1
            steps+=1

        second_memory.append_episodic()
        if second_memory.ready():
            #print("LEARNING")
            sample = second_memory.sample(sample_size = 10)
            training_steps += 1
            ok = agents.train(sample, training_steps)
            if ok:
                return
        episode += 1
        total_steps += steps
        # if memory.get_memory_real_size() >= 10:
        #     for i in range(10):
        #         # Sample from batch
        #         batch = memory.sample(100) # This should be different
        #         agents.learn(batch, episode)
        full_rwd += total_reward
        avg_rewd.append(full_rwd/episode)
        print(f'Reward: {total_reward} in {total_steps}, episode {episode} - AVG: {full_rwd/episode}')
    # Plot the rewards
    plt.plot(avg_rewd)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.show()

# Results: Avg: -64 after 750k
# Starting: Avg: -94

# 64 in 250-300 k
# 65 in 300k is fin


if __name__ == '__main__':
    run()