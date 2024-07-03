import pandas as pd

agent = 4
# Read CSV file into a pandas DataFrame
df = pd.read_csv(f'../blue_agent_{agent}.csv', header=None, names=['Net', 'proc', 'Acts'])

# Calculate overall action percentages
action_counts = df['Acts'].value_counts()
total_actions = len(df)
action_percentages = action_counts / total_actions * 100

# Print overall action percentages
print(f"Overall action percentages agent {agent}:")
print(action_percentages)
print("\n")

# Group by combination of 'Net' and 'proc' and count occurrences of each action
grouped = df.groupby(['Net', 'proc', 'Acts']).size().unstack(fill_value=0)

# Calculate percentages within each combination group
percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100

# Print results
print("Percentage of actions for each combination of (Net, proc):")
print(percentages)
