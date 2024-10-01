from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents.MAPPO.mappo import PPO
from CybORG.Agents.MAPPO.critic_network import CriticNetwork
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from statistics import mean, stdev
import numpy as np
import torch
import csv
from utils import save_statistics, save_agent_data_ppo, save_agent_network

class MAPPOEvaluator:
    EPISODE_LENGTH = 500
    MAX_EPS = 200

    def __init__(self, args):
        self.env = None
        self.agents = None
        self.centralized_critic = None
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.best_reward = -7000
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.agent_dict = {}
        for i in range(5):
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
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=self.EPISODE_LENGTH
        )
        cyborg = CybORG(scenario_generator=sg, seed=1)  # Add Seed
        env = BlueFlatWrapper(env=cyborg)
        env.reset()
        self.env = env
        self.agents = {
            f"blue_agent_{agent}": PPO(
                env.observation_space(f'blue_agent_{agent}').shape[0],
                len(env.get_action_space(f'blue_agent_{agent}')['actions']),
                self.MAX_EPS * self.EPISODE_LENGTH,
                agent,
                critic = None,
                messages= self.messages
            ) for agent in range(5)
        }
        print(f'Using agents {self.agents}')
        if self.load_best_network:
            for _, agent in self.agents.items():
                agent.load_network()
        if self.load_last_network:
            for _, agent in self.agents.items():
                agent.load_last_epoch()

    def run(self):
        self.initialize_environment()
        for i in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                actions_messages = {
                    agent_name: agent.get_action(
                        observations[agent_name],
                        state_value = torch.tensor(1, dtype=torch.int8)
                    )
                    for agent_name, agent in self.agents.items()
                    if agent_name in self.env.agents
                }
                actions = {agent_name: action for agent_name, (action, _) in actions_messages.items()}
                messages = {agent_name: message for agent_name, (_, message) in actions_messages.items()}
                for agent_name, _ in self.agents.items():
                    net, proc = self.extract_subnet_info(observations[agent_name], agent_name)
                    actions_total = self.env.get_action_space(agent_name)['actions']
                    #index = array_of_strings.index("Monitor") # 16 and 48
                    self.agent_dict[agent_name].append((net, proc, str(actions_total[actions[agent_name]]).split()[0]))
                # Perform action on the environment
                if self.messages:
                    observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    observations, reward, termination, truncation, _ = self.env.step(actions)
                # This terminates if all agents have 'termination=true'
                done = {
                    agent: termination.get(agent, False) or truncation.get(agent, False)
                    for agent in self.env.agents
                }
                if all(done.values()):
                    break
                r.append(mean(reward.values()))  # Add rewards
            self.total_rewards.append(sum(r))
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {mean(self.total_rewards)}")
            # Print average reward before rollout
        for agent_name, agent in self.agents.items():
            csv_filename = f"{agent_name}.csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Net', 'proc', 'Acts'])
                # Write data rows
                for data in self.agent_dict[agent_name]:
                    writer.writerow(data)
        save_statistics(self.total_rewards, self.average_rewards)

