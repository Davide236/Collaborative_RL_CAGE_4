from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.R_IPPO.ppo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev

import csv
import matplotlib.pyplot as plt


EPISODE_LENGTH = 500
MAX_EPS = 4000
LOAD_NETWORKS = False
LOAD_BEST = False
ROLLOUT = 10

def main():
    # Initialize CybORG environment
    sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                     green_agent_class=EnterpriseGreenAgent,
                                     red_agent_class=FiniteStateRedAgent,
                                     steps=EPISODE_LENGTH)
    cyborg = CybORG(scenario_generator=sg, seed=1) # Add Seed
    env = BlueFlatWrapper(env=cyborg)
    env.reset()
    # TODO: Check for 'Labels' and 'Mask' in the action space
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
    reward_before_update = 0
    best_reward = -2000
    average_rewards = []
    for i in range(MAX_EPS):
        # Reset the environment for each training episode
        observations, _ = env.reset()
        for agent_name, agent in agents.items():
            agent.set_initial_state(1)
        r = []
        for j in range(EPISODE_LENGTH): # Episode length
            count += 1
            for agent_name, agent in agents.items():
                agent.save_lstm_state()
            # Action selection for all agents
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name]
                )
                for agent_name, agent in agents.items() 
                if agent_name in env.agents
            }
            # Perform action on the environment
            observations, reward, termination, truncation, _ = env.step(actions)

            # Append the rewards and termination for each agent
            for agent_name, agent in agents.items():
                done = termination[agent_name] or truncation[agent_name]
                agent.episodic_rewards.append(reward[agent_name]) # Save reward
                agent.episodic_termination.append(done) # Save termination
            # This terminates if all agent have 'termination=true'
            done = {
                agent: termination.get(agent, False) or truncation.get(agent, False)
                for agent in env.agents
            }
            # If all agents are done (truncation) then end the episode
            if all(done.values()):
                break
            r.append(mean(reward.values())) # Add rewards  
        # Add to partial rewards  
        partial_rewards += sum(r)
        reward_before_update += sum(r)
        print(f"Final reward of the episode: {sum(r)}, length {count} - AVG: {partial_rewards/(i+1)}")
        total_rewards.append(sum(r)) 
        if (i+1) % ROLLOUT == 0:
            average_rewards.append(reward_before_update/ROLLOUT)
            if reward_before_update/ROLLOUT > best_reward:
                best_reward = reward_before_update/ROLLOUT
                for agent_name, agent in agents.items():
                    agent.save_network()
            reward_before_update = 0
        # Save rewards, state values and termination flags (divided per episodes)    
        for agent_name, agent in agents.items():
            agent.append_episodic()
            agent.clear_episodic()
            # Learn at every episode
            if (i+1) % ROLLOUT == 0:
                print(f"Policy update for  {agent_name}. Total steps: {count}")
                agent.learn(count) 
    # Graph of average rewards and print output results 
    for agent_name, agent in agents.items():
        agent.save_statistics_csv() 
        agent.save_last_epoch()
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