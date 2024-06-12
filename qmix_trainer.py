from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
from CybORG.Agents.QMIX.qmix import QMix
from CybORG.Agents.QMIX.buffer import ReplayBuffer
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
    agents = QMix(n_agents=n_agents, n_actions=agents_actions,obs_space=actor_dims,state_space=critic_dims, episode_length=EPISODE_LENGTH-1, total_episodes = EPISODE_LENGTH)
    memory = ReplayBuffer(1_000_000, actor_dims, batch_size=ROLLOUT, episode_length=EPISODE_LENGTH-1)

    return agents, memory

def transform_observations(obs):
    observations = []
    for i in range(5):
        observations.append(obs[f'blue_agent_{i}'])
    return observations

def main():
    # Initialize CybORG environment
    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                     green_agent_class=EnterpriseGreenAgent,
                                     red_agent_class=FiniteStateRedAgent,
                                     steps=EPISODE_LENGTH)
    cyborg = CybORG(scenario_generator=sg, seed=1) # Add Seed
    # Padding required for QMIX
    env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
    env.reset()
    agents, memory = setup_agents(env)
    training_steps = 0
    print(f'Using agents {agents}')
    total_rewards = [] 
    count = 0 # Keep track of total episodes
    partial_rewards = 0
    # average_rewards = []
    average_reward = 0
    for eps in range(MAX_EPS):
        # Reset the environment for each training episode
        observations, _ = env.reset()
        r = []
        for j in range(EPISODE_LENGTH): # Episode length
            count += 1
            # Action selection for all agents
            acts = agents.choose_actions(transform_observations(observations))
            actions = {
                f'blue_agent_{i}':acts[i]
                for i in range(5)
            }
            # Perform action on the environment
            new_observations, reward, termination, truncation, _ = env.step(actions)
            # Append the rewards and termination for each agent
            done = []
            for i in range(5):
                agent_name = f'blue_agent_{i}'
                done.append(termination[agent_name] or truncation[agent_name])
            obs1 = transform_observations(observations)
            obs2 = transform_observations(new_observations)
            reward2 = transform_observations(reward)
            # This terminates if all agent have 'termination=true'
            memory.store_episodic(obs1, acts, reward2, obs2, done, step=j)
            observations = new_observations
            #print(acts)
            if all(done):
                break
            r.append(mean(reward.values())) # Add rewards  
        partial_rewards += sum(r)
        print(f"Final reward of the episode: {sum(r)}, length {count} - AVG: {partial_rewards/(eps+1)}")
        # Add to partial rewards  
        total_rewards.append(sum(r))
        memory.append_episodic()
        if memory.ready():
            print("Training...")
            sample = memory.sample(ROLLOUT)
            training_steps += 1
            agents.train(sample, training_steps) 
        # Print average reward before rollout
        # if (eps+1) % ROLLOUT == 0:
        #     avg_rwd = partial_rewards/ROLLOUT
        #     average_rewards.append(avg_rwd)
        #     print(f"Average reward obtained before update: {avg_rwd}")
        #     # If the average reward is better than the best reward then save agents
        #     #agents.learn(memory) 
        #     partial_rewards = 0 
if __name__ == "__main__":
    main()