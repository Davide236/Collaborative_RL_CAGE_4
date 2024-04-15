import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files for each agent
csv_files = ['..\saved_statistics/data_agent_0.csv',
            '..\saved_statistics/data_agent_1.csv',
            '..\saved_statistics/data_agent_2.csv', 
            '..\saved_statistics/data_agent_3.csv',
            '..\saved_statistics/data_agent_4.csv']

# Read data from CSV files and store in a list
data = []
for file in csv_files:
    df = pd.read_csv(file)
    data.append(df)

# Print column names
for i, df in enumerate(data):
    print(f"Agent {i} column names:")
    print(df.columns)

# Plot Entropy
plt.figure(figsize=(10, 6))
for i, df in enumerate(data):
    plt.plot(df['Entropy'], label=f'Agent {i}')

plt.title('Entropy Evolution')
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)
plt.show()

# Plot Critic Loss
plt.figure(figsize=(10, 6))
for i, df in enumerate(data):
    plt.plot(df['Critic Loss'], label=f'Agent {i}')

plt.title('Critic Loss Evolution')
plt.xlabel('Time Step')
plt.ylabel('Critic Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Actor Loss
plt.figure(figsize=(10, 6))
for i, df in enumerate(data):
    plt.plot(df['Actor Loss'], label=f'Agent {i}')

plt.title('Actor Loss Evolution')
plt.xlabel('Time Step')
plt.ylabel('Actor Loss')
plt.legend()
plt.grid(True)
plt.show()

