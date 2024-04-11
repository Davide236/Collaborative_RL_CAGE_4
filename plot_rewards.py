import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('rewards.csv', header=None)

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
plt.plot(averages, label='Average Reward')

# Plot the area between the average curve plus one standard deviation and the average curve minus one standard deviation
plt.fill_between(range(len(averages)), np.array(averages) + np.array(std_deviations), np.array(averages) - np.array(std_deviations), alpha=0.3)

# Add labels and title
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.title('Average Reward with Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()
