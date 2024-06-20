import csv
import matplotlib.pyplot as plt
from statistics import mean, stdev
import torch


def save_agent_data(agents):
    for _, agent in agents.items():
        # Save to CSV
        data = zip(agent.entropy, agent.critic_loss, agent.actor_loss)
        with open(agent.save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Entropy', 'Critic Loss', 'Actor Loss'])  # Write header
            writer.writerows(data)

def save_agent_network(network, optimizer, path):
    print('Saving Networks and Optimizers.....')
    torch.save({
        'actor_state_dict': network.state_dict(),
        'actor_optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
def save_statistics(total_rewards, average_rewards):
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