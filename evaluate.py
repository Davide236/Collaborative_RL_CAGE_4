from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.PPO.ppo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
import numpy as np
import torch
import torch.nn as nn

import csv
import matplotlib.pyplot as plt


EPISODE_LENGTH = 500
MAX_EPS = 100
LOAD_NETWORKS = True
LOAD_BEST = False
ROLLOUT = 5

def main():
    # Initialize CybORG environment
    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                     green_agent_class=EnterpriseGreenAgent,
                                     red_agent_class=FiniteStateRedAgent,
                                     steps=EPISODE_LENGTH)
    cyborg = CybORG(scenario_generator=sg) # Add Seed
    env = BlueFlatWrapper(env=cyborg)
    env.reset()
    centralized_critic = 0 # Just for eval
    agents = {f"blue_agent_{agent}": PPO(env.observation_space(f'blue_agent_{agent}').shape[0], len(env.get_action_space(f'blue_agent_{agent}')['actions']), MAX_EPS*EPISODE_LENGTH, agent) for agent in range(5)}
    print(f'Using agents {agents}')
    if LOAD_NETWORKS:
        for agent_name, agent in agents.items():
            if LOAD_BEST:
                agent.load_network()
            else:
                agent.load_last_epoch()
    # TODO: Add recording of time
    total_rewards = [] 
    count = 0 # Keep track of total episodes
    partial_rewards = 0
    average_rewards = []
    for i in range(MAX_EPS):
        # Reset the environment for each training episode
        observations, _ = env.reset()
        r = []
        for j in range(EPISODE_LENGTH): # Episode length
            count += 1
            # Action selection for all agents
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name],
                    env.action_mask(agent_name)
                )
                for agent_name, agent in agents.items() 
                if agent_name in env.agents
            }
            # Perform action on the environment
            observations, reward, termination, truncation, _ = env.step(actions) #, messages=messages)
            # This terminates if all agent have 'termination=true'
            done = {
                agent: termination.get(agent, False) or truncation.get(agent, False)
                for agent in env.agents
            }
            # If all agents are done (truncation) then end the episode
            if all(done.values()):
                break
            r.append(mean(reward.values())) # Add rewards  
        print(f"Final reward of the episode: {sum(r)}, length {count}")
        # Add to partial rewards  
        partial_rewards += sum(r)
        total_rewards.append(sum(r)) 
    # Graph of average rewards and print output results 
    rewards_mean = mean(total_rewards)
    rewards_stdev = stdev(total_rewards)
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.show()
    print(f"Average reward: {rewards_mean}, standard deviation of {rewards_stdev}")

if __name__ == "__main__":
    main()
