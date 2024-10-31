import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ask for the number of files
num_files = int(input("Enter the number of reward files to load: "))

# Initialize lists to store data, averages, and rolling averages
data_list = []
averages_list = []
rolling_averages_list = []
labels = []

# Load each file dynamically and calculate averages (the files are loaded in order in rewards.csv files) [rewards.csv, rewards1.csv, rewards2.csv etc.]
for i in range(1, num_files + 1):
    file_name = f'../rewards{i}.csv' if i > 1 else '../rewards.csv'
    label = input(f"Enter label for file {file_name}: ")
    labels.append(label)
    
    # Load the CSV file
    data = pd.read_csv(file_name, header=None)
    data_list.append(data)
    
    # Calculate the averages
    total_sum = 0
    averages = []
    for j, value in enumerate(data[0], 1):
        total_sum += value
        average = total_sum / j
        averages.append(average)
    averages_list.append(averages)
    
    # Calculate the rolling average (50 episodes window)
    rolling_window = 25
    rolling_averages = [np.mean(data[0][k:k+rolling_window]) for k in range(0, len(data[0]), rolling_window)]
    rolling_averages_list.append(rolling_averages)

# Plot the average reward per episode for all files
#plt.figure(figsize=(10, 5))

# plt.subplot(2, 1, 1)
# for i in range(num_files):
#     plt.plot(averages_list[i], label=labels[i])
# plt.xlabel('Number of Episodes')
# plt.ylabel('Reward')
# plt.title('Average Reward')
# plt.legend()
# plt.grid(True)

# Plot the rolling average reward for all files
#plt.subplot(2, 1, 2)
plt.plot()
for i in range(num_files):
    plt.plot(rolling_averages_list[i], label=labels[i])
plt.xlabel('Number of 25-Episode Windows', fontsize=14)  # Set label fontsize
plt.ylabel('Reward', fontsize=17)  # Set label fontsize
#plt.title('Rolling Average Reward (50 episodes)', fontsize=14)  # Set title fontsize
plt.legend(fontsize=16)  # Set legend fontsize
plt.grid(True)

plt.tight_layout()
plt.show()
