import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('../rewards.csv', header=None)

# Calculate the average value and standard deviation at each timestep
averages = []
std_deviations = []
total_sum = 0
total_squared_sum = 0
for i, value in enumerate(data[0], 1):
    total_sum += value
    total_squared_sum += value ** 2
    average = total_sum / i
    variance = (total_squared_sum / i) - (average ** 2)
    std_deviation = np.sqrt(variance) if i > 1 else 0
    averages.append(average)
    std_deviations.append(std_deviation)

# Plot the average curve
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(averages, label='Average Reward')
plt.fill_between(range(len(averages)), np.array(averages) + np.array(std_deviations), np.array(averages) - np.array(std_deviations), alpha=0.3)
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.title('Average Reward with Standard Deviation')
plt.legend()
plt.grid(True)

# Calculate the average reward and standard deviation for every 50 episodes
rolling_window = 50
rolling_averages = [np.mean(data[0][i:i+rolling_window]) for i in range(0, len(data[0]), rolling_window)]
rolling_std_deviations = [np.std(data[0][i:i+rolling_window]) for i in range(0, len(data[0]), rolling_window)]

# Plot the rolling average curve
plt.subplot(2, 1, 2)
plt.plot(rolling_averages, label='Rolling Average Reward (50 episodes)')
plt.fill_between(range(len(rolling_averages)), np.array(rolling_averages) + np.array(rolling_std_deviations), np.array(rolling_averages) - np.array(rolling_std_deviations), alpha=0.3)
plt.xlabel('Number of 50-Episode Windows')
plt.ylabel('Reward')
plt.title('Rolling Average Reward with Standard Deviation (50 episodes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
