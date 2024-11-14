from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.IPPO.ippo import PPO
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean
import csv
from utils import save_statistics
import re

# Change
from collections import Counter

class IPPOEvaluator:
    EPISODE_LENGTH = 500
    MAX_EPS = 200
    def __init__(self, args):
        self.n_agents = 5
        self.agents = {}
        self.total_rewards = []
        self.average_rewards = []
        self.partial_rewards = 0
        self.count = 0  # Keep track of total episodes
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.agent_dict = {}
        for i in range(self.n_agents):
            self.agent_dict[f'blue_agent_{i}'] = []
        
    
    # Function to extract information for each subnet
    def extract_subnet_info(self, observation_vector, agent_number):
        total_subnets = 1
        # Agent 4 takes care of more subnets
        if agent_number == 'blue_agent_4':
            total_subnets = 3
        S = 9  # Number of subnets
        H = 16  # Maximum number of hosts in each subnet
        subnets_length = 3*S + 2*H
        subnet_info = []
        for i in range(total_subnets):
            subnet_start_index = i * (subnets_length) + 1
            subnet = observation_vector[subnet_start_index:subnet_start_index + subnets_length]
            subnet_vector = subnet[:S]
            blocked_subnets = subnet[S:2 * S]
            communication_policy = subnet[2 * S:3 * S]
            malicious_process_event_detected = subnet[3 * S:3 * S + H]
            malicious_network_event_detected = subnet[3 * S + H:]
            subnet_info.append({
                'subnet_vector': subnet_vector,
                'blocked_subnets': blocked_subnets,
                'communication_policy': communication_policy,
                'malicious_process_event_detected': malicious_process_event_detected,
                'malicious_network_event_detected': malicious_network_event_detected
            })
        malicious_network = []
        malicious_process = []
        total_network = 0
        total_process = 0
        # Iterate through each subnet information dictionary
        for subnet in subnet_info:
            # Append the 'malicious_network_event_detected' array to the malicious_network list'
            total_network += sum(subnet['malicious_network_event_detected'])
            total_process += sum(subnet['malicious_process_event_detected'])
            malicious_network.append(subnet['malicious_network_event_detected'])
            malicious_process.append(subnet['malicious_process_event_detected'])
        return total_network, total_process
            
    def initialize_environment(self):
        self.sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,
                                         green_agent_class=EnterpriseGreenAgent,
                                         red_agent_class=FiniteStateRedAgent,
                                         steps=self.EPISODE_LENGTH)
        self.cyborg = CybORG(scenario_generator=self.sg, seed=1)  # Add Seed
        env = BlueFlatWrapper(env=self.cyborg)
        env.reset()
        self.env = env
        self.agents = {f"blue_agent_{agent}": PPO(env.observation_space(f'blue_agent_{agent}').shape[0],
                                                  len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                                                  self.MAX_EPS*self.EPISODE_LENGTH, agent, self.messages) 
                       for agent in range(self.n_agents)}
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()
    
    def find_red_actions(self, red_actions, agent_name):
        number = re.findall(r'\d+', agent_name)
        #red_agent_original = f'red_agent_{int(number[0])}'
        red_agent = f'red_agent_{int(number[0])+1}'
        red_data = self.sg.get_red_agent_data_eval(red_agent)
        if not red_data:
            red_state = 'None'
        else:
            red_state = red_data[red_agent]
        if red_agent in red_actions:
            red_act = red_actions[red_agent]
            return red_act['action'], red_act['target'], red_state
        return 'None', 'None', red_state
    # Red 0: Contractor Network
    # Red 1: Restricted zone A - Blue 0
    # Red 2: Operational Zone A - Blue 1
    # Red 3: Restricted zone B - Blue 2
    # Red 4: Operational zone B - Blue 3
    # Red 5: Office network - Blue 4
    def run(self):
        self.initialize_environment()
        for _ in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                actions_messages = {
                    agent_name: agent.get_action(
                        observations[agent_name]
                    )
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }

                actions = {agent_name: action for agent_name, (action, _) in actions_messages.items()}
                messages = {agent_name: message for agent_name, (_, message) in actions_messages.items()}
                active_agents = self.env.active_agents
                red_actions = {}
                green_actions = {}
                for agent_name, _ in self.agents.items():
                    green_actions[agent_name] = [] 
                for agent in active_agents:
                    if 'red_agent' in agent:
                        red_action = str(self.env.get_last_action(agent))
                        red_action = red_action.replace('[','').replace(']','')
                        action = red_action.split()
                        action_name = action[0]
                        action_target = None
                        if len(action) > 1:
                            action_target = action[1]
                        red_actions[agent] = {'action':action_name, 'target': action_target }
                    if 'green_agent' in agent:
                        green_action = str(self.env.get_last_action(agent))
                        hostname = str(self.sg.green_agent_list_evaluation[agent])
                        if 'restricted_zone_a' in hostname:
                            green_actions['blue_agent_0'].append({'act':green_action, 'hostname': hostname})
                        elif 'operational_zone_a' in hostname:
                            green_actions['blue_agent_1'].append({'act':green_action, 'hostname': hostname})
                        elif 'restricted_zone_b' in hostname:
                            green_actions['blue_agent_2'].append({'act':green_action, 'hostname': hostname})
                        elif 'operational_zone_b' in hostname:
                            green_actions['blue_agent_3'].append({'act':green_action, 'hostname': hostname})
                        else:
                            green_actions['blue_agent_4'].append({'act':green_action, 'hostname': hostname})
                ip_map = self.cyborg.get_ip_map()
                inverted_ip_map = {str(v): k for k, v in ip_map.items()}
                for agent_name, _ in self.agents.items():
                    net, proc = self.extract_subnet_info(observations[agent_name], agent_name)
                    actions_total = self.env.get_action_space(agent_name)['actions']
                    labels = self.env.action_labels(agent_name)
                    action_label = labels[actions[agent_name]]
                    red_act, red_target, red_fsm = self.find_red_actions(red_actions, agent_name)
                    if red_fsm != 'None':
                        updated_hosts = [inverted_ip_map.get(host, host) for host in red_fsm['hosts']]
                        red_fsm['hosts'] = updated_hosts
                    if red_target != 'None':
                        red_target = inverted_ip_map.get(red_target)
                    #index = array_of_strings.index("Monitor") # 16 and 48
                    self.agent_dict[agent_name].append((net, proc, str(actions_total[actions[agent_name]]).split()[0], action_label, red_act, red_target, red_fsm, green_actions[agent_name]))
                # Perform action on the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)
                # This terminates if all agent have 'termination=true'
                done = {
                    agent: termination.get(agent, False) or truncation.get(agent, False)
                    for agent in self.env.agents
                }
                # If all agents are done (truncation) then end the episode
                if all(done.values()):
                    break
                r.append(mean(reward.values()))  # Add rewards  
            print(f"Final reward of the episode: {sum(r)}, length {self.count}")
            # Add to partial rewards  
            self.total_rewards.append(sum(r))
        
        for agent_name, agent in self.agents.items():
            csv_filename = f"{agent_name}.csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Net', 'proc', 'Blue_Acts','Blue_Acts_Extended', 'Red_acts', 'Red_acts_Target', 'Red_Fsm', 'Green_Acts'])
                # Write data rows
                for data in self.agent_dict[agent_name]:
                    writer.writerow(data)
        save_statistics(self.total_rewards, self.total_rewards)

