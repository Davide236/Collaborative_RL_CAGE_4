from ppo import PPO
import gym
import numpy
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

EPISODE_LENGTH = 500
MAX_EPS = 300

def main():
    env = gym.make('CartPole-v1')
    agent = PPO(env.observation_space.shape[0], env.action_space.n,50000)
    rewards = []
    learning_steps = 0
    count = 0
    i = 0
    learn_goal = 0
    # TODO: Check catastrofic forgetting after 60/70 k episodes
    while count < 200000:
        observation, _ = env.reset()
        total_reward = 0
        # Step 1: Reset Hidden State at each episode
        agent.set_initial_state(1)
        # This reset the hidden state to 0s (redoundant with the set_initial_state)
        terminal = torch.ones(1)
        for _ in range(EPISODE_LENGTH):
            learn_goal += 1
            count += 1
            # Step 1: Save current hidden state
            agent.save_lstm_state()
            # Step 3: Collect partial trajectories based on obs and hidden state
            action = agent.get_action(observation, terminal)
            observation, reward, terminated, truncated, _ = env.step(action)
            agent.save_rollout_data(reward, terminated)
            terminal = torch.tensor(terminated).float()
            total_reward += reward
            if terminated or truncated:
                break
        i += 1
        agent.append_episodic()
        rewards.append(total_reward)
        #print(f'Finished episode with total count: {count}, reward: {total_reward}, AVG: {numpy.array(rewards).mean()}')
        if learn_goal >= 500:
            # Print the average reward obtained before rollout!
            print(f"Learning in {learn_goal} steps. Total Count: {count} - AVG REW: {numpy.array(rewards).mean()}")
            learning_steps += 1
            agent.learn(count) 
            learn_goal = 0
    
    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

