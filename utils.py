import csv
import matplotlib.pyplot as plt
from statistics import mean, stdev
import torch

# Function which saves the loss data of QMIX agents
def save_agent_data_mixer(agents):
    data = zip(agents.loss)
    with open(agents.save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Agents Loss'])  # Write header
        writer.writerows(data)


# Function which saves the loss data of MADDPG agents
def save_agent_data_maddpg(agents):
    for agent in agents.agents:
        data = zip(agent.critic_loss, agent.actor_loss)
        with open(agent.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Critic Loss', 'Actor Loss'])  # Write header
            writer.writerows(data)

# Function which saves the loss data of PPO agents
def save_agent_data_ppo(agents):
    for _, agent in agents.items():
        # Save to CSV
        data = zip(agent.entropy, agent.critic_loss, agent.actor_loss)
        with open(agent.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entropy', 'Critic Loss', 'Actor Loss'])  # Write header
            writer.writerows(data)

# Function which saves the network of a given agent in the given path
def save_agent_network(network, optimizer, path):
    print('Saving Networks and Optimizers.....')
    torch.save({
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Function used to save the statistics relative to the training run of the agents  
def save_statistics(total_rewards, average_rewards):
    rewards_mean = mean(total_rewards)
    rewards_stdev = stdev(total_rewards)
    total_rewards_transposed = [[elem] for elem in average_rewards]
    
    # Save the training reward history of the agents
    with open('reward_history.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Rewards'])  # Write header
        writer.writerows(total_rewards_transposed)
    
    # Show rewards on plot and display mean and standard deviation
    plt.ion()  # Turn on interactive mode
    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.savefig('final_training.png')
    print(f"Average reward: {rewards_mean}, standard deviation of {rewards_stdev}")

# Class used to normalize the reward obtained by the agents during training
class RewardNormalizer:
    # Low minimum value, approximating the minimum reward that can be achieved per each episode
    def __init__(self, min_value=-35, max_value=0):
        self.min_value = min_value
        self.max_value = max_value
    
    def normalize(self, reward):
        # Ensure the reward is within the expected range
        if reward > self.max_value:
            reward = self.max_value
        elif reward < self.min_value:
            reward = self.min_value

        # Normalize the reward to the range [0, 1]
        normalized_reward = (reward - self.min_value) / (self.max_value - self.min_value)
        return normalized_reward