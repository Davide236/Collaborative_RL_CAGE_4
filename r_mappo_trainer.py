from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.R_MAPPO.r_mappo import PPO
from CybORG.Agents.R_MAPPO.critic_network import CriticNetwork
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
import numpy as np
import torch
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt


EPISODE_LENGTH = 500
MAX_EPS = 4000
LOAD_NETWORKS = False
LOAD_BEST = False
ROLLOUT = 2

def concatenate_observations(observations, agents):
    observation_list = []
    for agent_name, agent in agents.items():
        observation_list.extend(observations[agent_name])
    normalized_state = (observation_list - np.mean(observation_list)) / (np.std(observation_list) + 1e-8)
    state = torch.FloatTensor(normalized_state.reshape(1,-1))
    return state

def save_last_epoch(critic,checkpoint):
    print('Saving Networks.....')
    torch.save(critic.state_dict(),checkpoint)
    
# Save both actor and critic networks of the agent
def save_network(critic,checkpoint):
    print('Saving Networks.....')
    torch.save(critic.state_dict(),checkpoint)
    
    # Initialize checkpoint to save the different agents
def init_checkpoint():
    checkpoint_file_critic = os.path.join('saved_networks', f'r_critic_mappoppo_central')
    last_checkpoint_file_critic = os.path.join('last_networks', f'r_critic_mappo_central')
    return checkpoint_file_critic, last_checkpoint_file_critic

def main():
    # Initialize CybORG environment
    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                     green_agent_class=EnterpriseGreenAgent,
                                     red_agent_class=FiniteStateRedAgent,
                                     steps=EPISODE_LENGTH)
    cyborg = CybORG(scenario_generator=sg, seed=1) # Add Seed
    env = BlueFlatWrapper(env=cyborg)
    env.reset()
    best_critic, last_critic = init_checkpoint()
    lr = 2.5e-4 # Learning rate of optimizer
    eps = 1e-5
    centralized_critic = CriticNetwork(env.observation_space('blue_agent_4').shape[0],env.observation_space('blue_agent_0').shape[0], 5)
    critic_optimizer = torch.optim.Adam(centralized_critic.parameters(), lr=lr)
    agents = {f"blue_agent_{agent}": PPO(env.observation_space(f'blue_agent_{agent}').shape[0], len(env.get_action_space(f'blue_agent_{agent}')['actions']), MAX_EPS*EPISODE_LENGTH, agent, centralized_critic, critic_optimizer) for agent in range(5)}
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
    best_reward = -2000
    average_rewards = []
    for i in range(MAX_EPS):
        # Reset the environment for each training episode
        observations, _ = env.reset()
        for agent_name, agent in agents.items():
            agent.set_initial_state(1)
        centralized_critic.get_init_state(1)
        r = []
        for j in range(EPISODE_LENGTH): # Episode length
            count += 1
            observations_list = concatenate_observations(observations, agents)
            state_value = centralized_critic(observations_list)
            # Action selection for all agents
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name],
                    state_value
                )
                for agent_name, agent in agents.items() 
                if agent_name in env.agents
            }
            # Perform action on the environment
            observations, reward, termination, truncation, _ = env.step(actions) #, messages=messages)

            # Append the rewards and termination for each agent
            for agent_name, agent in agents.items():
                done = termination[agent_name] or truncation[agent_name]
                agent.memory.save_end_episode(reward[agent_name], done, observations_list)
            # This terminates if all agent have 'termination=true'
            done = {
                agent: termination.get(agent, False) or truncation.get(agent, False)
                for agent in env.agents
            }
            #for agent in env.agents:
                #continue
                #print(f"Agent: {agent} made action: {env.get_last_action(agent)}")
            # If all agents are done (truncation) then end the episode
            if all(done.values()):
                break
            r.append(mean(reward.values())) # Add rewards  
        # Add to partial rewards  
        partial_rewards += sum(r)
        total_rewards.append(sum(r))
        print(f"Final reward of the episode: {sum(r)}, length {count} - AVG: {mean(total_rewards)}")
        # Print average reward before rollout
        if (i+1) % ROLLOUT == 0:
            avg_rwd = partial_rewards/ROLLOUT
            average_rewards.append(avg_rwd)
            print(f"Average reward obtained before update: {avg_rwd}")
            # If the average reward is better than the best reward then save agents
            if avg_rwd > best_reward:
                best_reward = avg_rwd
                for agent_name, agent in agents.items():
                    agent.save_network()
                    save_network(centralized_critic, best_critic)
            partial_rewards = 0 
        # Save rewards, state values and termination flags (divided per episodes)    
        for agent_name, agent in agents.items():
            agent.memory.append_episodic()
            # Every 5 episodes perform a policy update
            if (i+1) % ROLLOUT == 0:
                print(f"Policy update for  {agent_name}. Total steps: {count}")
                agent.learn(count) 
    # Save loss data
    for agent_name, agent in agents.items():
        agent.save_statistics_csv() 
        agent.save_last_epoch() 
        save_last_epoch(centralized_critic, last_critic)
    # Graph of average rewards and print output results 
    rewards_mean = mean(total_rewards)
    rewards_stdev = stdev(total_rewards)
    total_rewards_transposed = [[elem] for elem in average_rewards]
    with open('reward_history.csv', mode='w', newline='') as file:
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
