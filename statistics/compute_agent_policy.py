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
    df = pd.read_csv(file_path)
    
    results = pd.DataFrame(columns=['Red_acts', 'Acts'])

    # Loop through the DataFrame
    for index, row in df.iterrows():
        red_act = row['Red_acts']
        if red_act in steps_after:
            step = steps_after[red_act]
            if index + step < len(df):
                act_after = df.at[index + step, 'Acts']
                results = pd.concat([results, pd.DataFrame({'Red_acts': [red_act], 'Acts': [act_after]})], ignore_index=True)
    
    # Group by 'Red_acts' and count the occurrences of each 'Acts'
    grouped = results.groupby(['Red_acts', 'Acts']).size().reset_index(name='count')
    
    # Calculate the total count for each 'Red_acts' to compute percentages
    total_counts = results['Red_acts'].value_counts().reset_index(name='total_count').rename(columns={'index': 'Red_acts'})
    
    # Merge total counts with grouped data
    grouped = grouped.merge(total_counts, on='Red_acts')
    
    grouped['percentage'] = (grouped['count'] / grouped['total_count']) * 100
    
    # For each 'Red_acts', find the most frequent 'Acts'
    most_frequent_acts = grouped.loc[grouped.groupby('Red_acts')['count'].idxmax()]
    
    # Print results
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

