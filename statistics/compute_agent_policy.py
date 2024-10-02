import pandas as pd
import ast  # To safely parse the 'Green_Acts' and 'Red_Fsm' columns which contain list-like strings
import re  # Regular expressions to extract action type from Green_Acts
from collections import defaultdict

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

def extract_action_type(action_str):
    """
    Extract the action type from the Green_Acts string by stripping out
    the parameters (IP addresses, etc.) and returning only the base action.
    E.g., '[GreenLocalWork 10.0.169.196]' becomes 'GreenLocalWork'.
    """
    match = re.match(r"\[(\w+)", action_str)  # Match the first word inside square brackets
    if match:
        return match.group(1)  # Return the action type (e.g., 'GreenLocalWork')
    return None

def analyze_csv_with_steps(file_path, steps_after):
    # Read the CSV file and rename 'Acts' to 'Blue_Acts'
    df = pd.read_csv(file_path).rename(columns={'Acts': 'Blue_Acts'})
    
    # Analyze the most chosen Blue actions overall
    blue_action_counts = df['Blue_Acts'].value_counts().reset_index(name='count').rename(columns={'index': 'Blue_Acts'})
    
    # Calculate the total number of Blue actions for percentage calculation
    total_blue_actions = blue_action_counts['count'].sum()
    
    # Add percentage column to blue_action_counts DataFrame
    blue_action_counts['percentage'] = (blue_action_counts['count'] / total_blue_actions) * 100
    
    # Initialize a list to store Blue/Green and Blue/Red correlations
    blue_to_green_correlation = []
    blue_to_red_correlation = []
    
    # To track the host states and blue agent actions per state
    host_state_machine = {}
    
    # To track the global state actions across all hosts
    global_state_actions = defaultdict(list)

    # Analyze correlation with Red_Acts and Blue_Acts
    for red_act, step in steps_after.items():
        step += 1
        mask = df['Red_acts'] == red_act
        indices = df.index[mask]
        valid_indices = indices[indices + step < len(df)]
        actions_after = df.loc[valid_indices + step, 'Blue_Acts'].values
        blue_to_red_correlation.extend(zip([red_act] * len(actions_after), actions_after))

    # Convert Red/Blue correlation results to DataFrame
    red_blue_df = pd.DataFrame(blue_to_red_correlation, columns=['Red_acts', 'Blue_Acts'])
    
    # Group by 'Red_acts' and count the occurrences of each 'Blue_Acts'
    grouped_red_blue = red_blue_df.groupby(['Red_acts', 'Blue_Acts']).size().reset_index(name='count')
    
    # Calculate the total count for each 'Red_acts' to compute percentages
    total_counts_red_blue = red_blue_df['Red_acts'].value_counts().reset_index(name='total_count').rename(columns={'index': 'Red_acts'})
    
    # Merge total counts with grouped Red/Blue data
    grouped_red_blue = grouped_red_blue.merge(total_counts_red_blue, on='Red_acts')
    
    # Calculate the percentage for Red/Blue actions correlation
    grouped_red_blue['percentage'] = (grouped_red_blue['count'] / grouped_red_blue['total_count']) * 100
    
    # Sort the grouped data by percentage in descending order
    grouped_red_blue = grouped_red_blue.sort_values(by=['Red_acts', 'percentage'], ascending=[True, False])

    # Analyze Blue/Green correlation by checking Green actions before Blue actions
    df['Green_Acts_Parsed'] = df['Green_Acts'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    
    for idx in range(1, len(df)):
        blue_row = df.iloc[idx]
        prev_green_row = df.iloc[idx - 1]
        
        blue_extended = blue_row['Blue_Acts_Extended']
        prev_green_acts = prev_green_row['Green_Acts_Parsed']
        
        # Correlate Blue actions with the corresponding Green action right before
        if prev_green_acts:
            for green_act in prev_green_acts:
                green_host = green_act['hostname']
                green_action_type = extract_action_type(green_act['act'])
                
                # Check if the Blue action is related to the same host
                if pd.notna(blue_extended) and green_host in blue_extended:
                    blue_to_green_correlation.append({
                        'Blue_Action': blue_row['Blue_Acts'],
                        'Green_Action': green_action_type,
                        'Green_Host': green_host
                    })
    
    # Convert Blue/Green correlation to DataFrame
    blue_green_df = pd.DataFrame(blue_to_green_correlation)
    
    # Group by 'Blue_Action' and 'Green_Action' and count occurrences
    grouped_green_blue = blue_green_df.groupby(['Blue_Action', 'Green_Action']).size().reset_index(name='count')
    
    # Calculate the total count for each 'Blue_Action' to compute percentages
    total_counts_green_blue = blue_green_df['Blue_Action'].value_counts().reset_index(name='total_count').rename(columns={'index': 'Blue_Action'})
    
    # Merge total counts with grouped Blue/Green data
    grouped_green_blue = grouped_green_blue.merge(total_counts_green_blue, on='Blue_Action')
    
    # Calculate the percentage for Blue/Green actions correlation
    grouped_green_blue['percentage'] = (grouped_green_blue['count'] / grouped_green_blue['total_count']) * 100
    
    # Sort the grouped Blue/Green data by percentage in descending order
    grouped_green_blue = grouped_green_blue.sort_values(by=['Blue_Action', 'percentage'], ascending=[True, False])

    # Track the current state for each host
    current_host_state = {}

    # Loop through the DataFrame to find Green actions that occur before Blue actions
    for idx in range(len(df)):
        blue_row = df.iloc[idx]
        blue_extended = blue_row['Blue_Acts_Extended']
        red_fsm = blue_row['Red_Fsm']
        
        # Update the host's state from Red_Fsm, if available
        if pd.notna(red_fsm) and red_fsm != 'None':
            fsm_data = ast.literal_eval(red_fsm)  # Safely parse the Red_Fsm data
            hosts = fsm_data.get('hosts', [])
            states = fsm_data.get('state', [])
            
            for host, state in zip(hosts, states):
                # Ignore hosts with IPs, only track hostnames
                if not re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host):  # Match if it's an IP address
                    current_host_state[host] = state  # Set the current state for the host
                    # Initialize the host in state machine if not already done
                    if host not in host_state_machine:
                        host_state_machine[host] = {'current_state': state, 'actions': {state: []}}
                    elif host_state_machine[host]['current_state'] != state:
                        # If transitioning to a new state, initialize the new state if not present
                        if state not in host_state_machine[host]['actions']:
                            host_state_machine[host]['actions'][state] = []
                        # Update the host's current state
                        host_state_machine[host]['current_state'] = state

        # If there is a Blue action with a target host, track it under the host's current state
        if pd.notna(blue_extended):
            for host in current_host_state:
                if host in blue_extended:
                    current_state = host_state_machine[host]['current_state']
                    host_state_machine[host]['actions'][current_state].append(blue_row['Blue_Acts'])
                    
                    # Also track globally by state
                    global_state_actions[current_state].append(blue_row['Blue_Acts'])

    # Print Red/Blue correlations
    print("Correlation between Red_Acts and Blue_Acts:")
    for red_act in grouped_red_blue['Red_acts'].unique():
        print(f"For Red_acts '{red_act}':")
        red_act_group = grouped_red_blue[grouped_red_blue['Red_acts'] == red_act]
        for _, row in red_act_group.iterrows():
            print(f" - Blue_Act '{row['Blue_Acts']}' occurred {row['count']} times ({row['percentage']:.2f}%)")

    print("\n" + "="*50 + "\n")

    # Print Blue/Green correlations
    print("Correlation between Blue_Acts and Green_Acts:")
    for blue_action in grouped_green_blue['Blue_Action'].unique():
        print(f"For Blue_Action '{blue_action}':")
        blue_action_group = grouped_green_blue[grouped_green_blue['Blue_Action'] == blue_action]
        for _, row in blue_action_group.iterrows():
            print(f" - Green_Action '{row['Green_Action']}' occurred {row['count']} times ({row['percentage']:.2f}%)")
    
    print("\n" + "="*50 + "\n")

    # Print the state machine for each host
    print("State machine for each host (Blue agent actions per state):")
    for host, machine in host_state_machine.items():
        print(f"Host: {host}")
        for state, actions in machine['actions'].items():
            action_counts = pd.Series(actions).value_counts()
            total_actions = len(actions)
            print(f" - State '{state}':")
            for action, count in action_counts.items():
                print(f"   - Action '{action}' occurred {count} times ({(count / total_actions) * 100:.2f}%)")
    
    print("\n" + "="*50 + "\n")

    # Now, calculate and print the global state actions across hosts
    print("Global actions by state across all hosts:")
    for state, actions in global_state_actions.items():
        if actions:
            action_counts = pd.Series(actions).value_counts()
            total_actions = len(actions)
            print(f"State '{state}':")
            for action, count in action_counts.items():
                print(f" - Action '{action}' occurred {count} times ({(count / total_actions) * 100:.2f}%)")
    
    print("\nAnalysis complete.\n")

# Compute the policy of the given agent
agent = int(input("Enter agent number: "))
file_path = f'../blue_agent_{agent}.csv'
analyze_csv_with_steps(file_path, steps_after)
