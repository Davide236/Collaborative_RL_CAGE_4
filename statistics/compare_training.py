import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the first CSV file
data1 = pd.read_csv('../rewards.csv', header=None)
data2 = pd.read_csv('../rewards2.csv', header=None)
data3 = pd.read_csv('../rewards3.csv', header=None)

# Calculate the average value for each file
def calculate_averages(data):
    averages = []
    total_sum = 0
    for i, value in enumerate(data[0], 1):
        total_sum += value
        average = total_sum / i
        averages.append(average)
    return averages

averages1 = calculate_averages(data1)
averages2 = calculate_averages(data2)
averages3 = calculate_averages(data3)

# Plot the average reward per episode for both files
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(averages1, label='R-IPPO', color='blue')
plt.plot(averages2, label='MAPPO', color='orange')
plt.plot(averages3, label='IPPO', color='green')
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.title('Average Reward')
plt.legend()
plt.grid(True)

# Calculate the average reward for every 50 episodes for both files
rolling_window = 50
rolling_averages1 = [np.mean(data1[0][i:i+rolling_window]) for i in range(0, len(data1[0]), rolling_window)]
rolling_averages2 = [np.mean(data2[0][i:i+rolling_window]) for i in range(0, len(data2[0]), rolling_window)]
rolling_averages3 = [np.mean(data3[0][i:i+rolling_window]) for i in range(0, len(data3[0]), rolling_window)]

# Plot the rolling average reward for both files
plt.subplot(2, 1, 2)
plt.plot(rolling_averages1, label='R-IPPO', color='blue')
plt.plot(rolling_averages2, label='MAPPO', color='orange')
plt.plot(rolling_averages3, label='IPPO', color='green')
plt.xlabel('Number of 50-Episode Windows')
plt.ylabel('Reward')
plt.title('Rolling Average Reward (50 episodes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
