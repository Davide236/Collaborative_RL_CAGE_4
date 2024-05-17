from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
from CybORG.Agents.MADDPG.maddpg import MADDPG
from CybORG.Agents.MADDPG.replay_buffer import MultiAgentReplayBuffer
import numpy as np

import csv
import matplotlib.pyplot as plt


EPISODE_LENGTH = 500
MAX_EPS = 1000
LOAD_NETWORKS = False
LOAD_BEST = False
ROLLOUT = 5

def setup_agents(env):
    n_agents = 5
    actor_dims = []
    agents_actions = []
    # Dimension of observation_space (for each agent)
    for agent in range(n_agents):
        actor_dims.append(env.observation_space(f'blue_agent_{agent}').shape[0])
        agents_actions.append(len(env.get_action_space(f'blue_agent_{agent}')['actions']))
    critic_dims = sum(actor_dims)
    agents = MADDPG(actor_dims, critic_dims, n_agents, agents_actions)
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        agents_actions, n_agents, batch_size=3000)
    return agents, memory

def transform_observations(obs):
    observations = []
    for i in range(5):
        observations.append(obs[f'blue_agent_{i}'])
    return observations

def concatenate_observations(observations):
    observations_list = []
    for i in range(5):
        agent_name = f'blue_agent_{i}'
        observations_list.extend(observations[agent_name])
    # normalized_state = (observations_list - np.mean(observations_list)) / (np.std(observations_list) + 1e-8)
    # state = torch.FloatTensor(normalized_state.reshape(1,-1))
    return observations_list

def main():
    # Initialize CybORG environment
    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                     green_agent_class=EnterpriseGreenAgent,
                                     red_agent_class=FiniteStateRedAgent,
                                     steps=EPISODE_LENGTH)
    cyborg = CybORG(scenario_generator=sg, seed=1) # Add Seed
    env = BlueFlatWrapper(env=cyborg)
    env.reset()
    agents, memory = setup_agents(env)
    
    print(f'Using agents {agents}')
    total_rewards = [] 
    count = 0 # Keep track of total episodes
    partial_rewards = 0
    average_rewards = []
    average_reward = 0
    for eps in range(MAX_EPS):
        # Reset the environment for each training episode
        observations, _ = env.reset()
        r = []
        for j in range(EPISODE_LENGTH): # Episode length
            count += 1
            # Action selection for all agents
            acts = agents.choose_actions(observations, evaluate=False)
            actions = {
                f'blue_agent_{i}':acts[i]
                for i in range(5)
            }
            old_central_observations = concatenate_observations(observations)
            # Perform action on the environment
            new_observations, reward, termination, truncation, _ = env.step(actions)
            new_central_observations = concatenate_observations(new_observations)
            # Append the rewards and termination for each agent
            done = []
            for i in range(5):
                agent_name = f'blue_agent_{i}'
                done.append(termination[agent_name] or truncation[agent_name])
            obs1 = transform_observations(observations)
            obs2 = transform_observations(new_observations)
            reward2 = transform_observations(reward)
            # This terminates if all agent have 'termination=true'
            memory.store_transition(obs1, old_central_observations, acts, reward2, obs2, new_central_observations, done)
            observations = new_observations
            if all(done):
                break
            r.append(mean(reward.values())) # Add rewards  
        partial_rewards += sum(r)
        average_rewards.append(sum(r))
        print(f"Final reward of the episode: {sum(r)}, length {count} - AVG: {partial_rewards/(eps+1)}")
        # Add to partial rewards  
        total_rewards.append(sum(r))
        agents.learn(memory) 
        # Print average reward before rollout
        # if (eps+1) % ROLLOUT == 0:
        #     avg_rwd = partial_rewards/ROLLOUT
        #     average_rewards.append(avg_rwd)
        #     print(f"Average reward obtained before update: {avg_rwd}")
        #     # If the average reward is better than the best reward then save agents
        #     #agents.learn(memory) 
        #     partial_rewards = 0 
    rewards_mean = mean(total_rewards)
    rewards_stdev = stdev(total_rewards)
    total_rewards_transposed = [[elem] for elem in average_rewards]
    with open('maddpg_reward_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rewards'])  # Write header
        writer.writerows(total_rewards_transposed)
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.show()
    print(f"Average reward: {rewards_mean}, standard deviation of {rewards_stdev}")
if __name__ == "__main__":
    main()