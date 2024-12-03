import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For smoothing

# Input: Number of files
num_files = int(input("Enter the number of reward files to load: "))

# Initialize data structures
data_list = []
labels = []

# Load files and calculate means and error bars
mean_list = []
error_list = []
lower_bound_list = []
upper_bound_list = []

for i in range(1, num_files + 1):
    # Dynamic file naming
    file_name = f'../Rewards{i}.csv' if i > 1 else '../Rewards1.csv'
    label = input(f"Enter label for file {file_name}: ")
    labels.append(label)
    
    # Load the CSV file
    data = pd.read_csv(file_name, header=None)
    data_list.append(data)
    
    # Calculate mean and standard error for each row
    mean_values = data.mean(axis=1)  # Mean across columns for each row
    error_values = data.std(axis=1) / np.sqrt(data.shape[1])  # Standard error
    
    # Smooth the mean values and error bounds using Savitzky-Golay filter
    smoothed_mean = savgol_filter(mean_values, window_length=51, polyorder=3)  # Smooth mean
    smoothed_error = savgol_filter(error_values, window_length=51, polyorder=3)  # Smooth error
    
    mean_list.append(smoothed_mean)
    error_list.append(smoothed_error)
    
    # Calculate smoothed upper and lower bounds
    lower_bound = smoothed_mean - smoothed_error
    upper_bound = smoothed_mean + smoothed_error
    lower_bound_list.append(lower_bound)
    upper_bound_list.append(upper_bound)

# Plotting
plt.figure(figsize=(10, 6))

colors = ['blue', 'orange', 'green', 'red']  # Add more colors if needed
markers = ['o', 's', 'x', '^']  # Marker styles for each line

for i in range(num_files):
    # Calculate the average of the smoothed error for the label
    avg_error = np.mean(error_list[i])
    label_with_error = f"{labels[i]} (± {avg_error:.2f})"  # Append error value to the label
    
    # Plot smoothed mean with shaded error margins
    plt.plot(
        mean_list[i],
        label=label_with_error,
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        markevery=200,  # Place markers at regular intervals
        linestyle='-',  # Solid line
        linewidth=1.5
    )
    plt.fill_between(
        range(len(mean_list[i])),
        lower_bound_list[i],  # Smoothed lower bound
        upper_bound_list[i],  # Smoothed upper bound
        color=colors[i % len(colors)],
        alpha=0.2  # Transparency for shading
    )

# Add labels, title, and grid
plt.xlabel('Number of 20-Episode Windows', fontsize=12, fontweight='bold')
plt.ylabel('Mean Reward', fontsize=12, fontweight='bold')
#plt.title('Smoothed Mean Reward with Error Margins', fontsize=14, fontweight='bold')
plt.legend(title="Method", loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Tight layout and show
plt.tight_layout()
plt.show()
