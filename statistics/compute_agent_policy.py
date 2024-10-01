import pandas as pd

# Define the steps after for each Red_acts
steps_after = {
    "DiscoverRemoteSystems": 1,
    "AggressiveServiceDiscovery": 1,
    "StealthServiceDiscovery": 3,
    "PrivilegeEscalate": 2,
    "Impact": 2,
    "DegradeServices": 2,
    "DiscoverDeception": 2,
    "Sleep": 1,
    "None": 1,
    "ExploitRemoteService": 4
}

def analyze_csv_with_steps(file_path, steps_after):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Analyze the most chosen Blue actions overall
    blue_action_counts = df['Acts'].value_counts().reset_index(name='count').rename(columns={'index': 'Acts'})
    
    # Calculate the total number of Blue actions for percentage calculation
    total_blue_actions = blue_action_counts['count'].sum()
    
    # Add percentage column to blue_action_counts DataFrame
    blue_action_counts['percentage'] = (blue_action_counts['count'] / total_blue_actions) * 100
    
    # Print the most chosen Blue actions with percentages
    print("Most chosen Blue actions overall:")
    for _, row in blue_action_counts.iterrows():
        print(f" - Action '{row['Acts']}' chosen {row['count']} times ({row['percentage']:.2f}%)")
    print("\n" + "="*50 + "\n")
    
    results = []
    
    # Loop through the DataFrame using vectorized operations
    for red_act, step in steps_after.items():
        step += 1
        mask = df['Red_acts'] == red_act
        indices = df.index[mask]
        valid_indices = indices[indices + step < len(df)]
        actions_after = df.loc[valid_indices + step, 'Acts'].values
        results.extend(zip([red_act] * len(actions_after), actions_after))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results, columns=['Red_acts', 'Acts'])
    
    # Group by 'Red_acts' and count the occurrences of each 'Acts'
    grouped = results_df.groupby(['Red_acts', 'Acts']).size().reset_index(name='count')
    
    # Calculate the total count for each 'Red_acts' to compute percentages
    total_counts = results_df['Red_acts'].value_counts().reset_index(name='total_count').rename(columns={'index': 'Red_acts'})
    
    # Merge total counts with grouped data
    grouped = grouped.merge(total_counts, on='Red_acts')
    
    # Calculate the percentage
    grouped['percentage'] = (grouped['count'] / grouped['total_count']) * 100
    
    # Sort the grouped data by percentage in descending order
    grouped = grouped.sort_values(by=['Red_acts', 'percentage'], ascending=[True, False])
    
    # For each 'Red_acts', find the most frequent 'Acts'
    most_frequent_acts = grouped.loc[grouped.groupby('Red_acts')['count'].idxmax()]
    
    # Print the results
    for red_act in grouped['Red_acts'].unique():
        print(f"For Red_acts '{red_act}':")
        
        red_act_group = grouped[grouped['Red_acts'] == red_act]
        for _, row in red_act_group.iterrows():
            print(f" - Acts '{row['Acts']}' appears {row['count']} times ({row['percentage']:.2f}%)")
        
        most_frequent_act = most_frequent_acts[most_frequent_acts['Red_acts'] == red_act]
        print(f"Most frequent Acts for '{red_act}' is '{most_frequent_act['Acts'].values[0]}' with {most_frequent_act['count'].values[0]} occurrences.\n")

# Compute the policy of the given agent
agent = int(input("Enter agent number: "))
file_path = f'../blue_agent_{agent}.csv'
analyze_csv_with_steps(file_path, steps_after)

