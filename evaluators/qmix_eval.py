from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Agents.QMIX.qmix import QMix
from statistics import mean
import csv
import matplotlib.pyplot as plt
from utils import save_statistics
import re


class QMIXEvaluator:
    EPISODE_LENGTH = 500
    MAX_EPS = 500
    def __init__(self, args):
        self.env = None
        self.agents = None
        self.n_agents = 5
        self.total_rewards = []
        self.partial_rewards = 0
        self.average_rewards = []
        self.count = 0  # Keep track of total episodes
        self.training_steps = 0
        self.best_reward = -7000
        self.load_last_network = args.Load_last
        self.load_best_network = args.Load_best
        self.messages = args.Messages
        self.agent_dict = {}
        for i in range(self.n_agents):
            self.agent_dict[f'blue_agent_{i}'] = []

    def setup_agents(self, env):
        n_agents = 5
        actor_dims = []
        agents_actions = []
        # Dimension of observation_space (for each agent)
        for agent in range(n_agents):
            actor_dims.append(env.observation_space(f'blue_agent_{agent}').shape[0])
            agents_actions.append(len(env.get_action_space(f'blue_agent_{agent}')['actions']))
        critic_dims = sum(actor_dims)
        agents = QMix(
            n_agents=n_agents,
            n_actions=agents_actions,
            obs_space=actor_dims,
            state_space=critic_dims,
            episode_length=self.EPISODE_LENGTH - 1,
            total_episodes=self.EPISODE_LENGTH,
            messages = self.messages
        )
        return agents

    @staticmethod
    def transform_observations(obs):
        observations = []
        for i in range(5):
            observations.append(obs[f'blue_agent_{i}'])
        return observations
    
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
        # Padding required for QMIX
        env = BlueFlatWrapper(env=cyborg, pad_spaces=True)
        env.reset()
        self.env = env
        self.agents = self.setup_agents(env)
        # TODO: Change this
        self.load_last_network = False
        if self.load_best_network:
            self.agents.load_network()
        if self.load_last_network:
            self.agents.load_last_epoch()
    
    def find_red_actions(self, red_actions, agent_name):
        number = re.findall(r'\d+', agent_name)
        red_agent = f'red_agent_{int(number[0])+1}'
        if red_agent in red_actions:
            red_act = red_actions[red_agent]
            return red_act['action']
        return 'None'
    
    def run(self):
        self.initialize_environment()
        for eps in range(self.MAX_EPS):
            # Reset the environment for each training episode
            observations, _ = self.env.reset()
            r = []
            for j in range(self.EPISODE_LENGTH):  # Episode length
                self.count += 1
                # Action selection for all agents
                acts, msg = self.agents.choose_actions(self.transform_observations(observations))
                actions = {
                    f'blue_agent_{i}': acts[i]
                    for i in range(5)
                }
                active_agents = self.env.active_agents
                red_actions = {}
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
                for agent_idx in range(self.n_agents):
                    agent_name = f'blue_agent_{agent_idx}'
                    net, proc = self.extract_subnet_info(observations[agent_name], agent_name)
                    actions_total = self.env.get_action_space(agent_name)['actions']
                    red_act = self.find_red_actions(red_actions, agent_name)
                    #index = array_of_strings.index("Monitor") # 16 and 48
                    self.agent_dict[agent_name].append((net, proc, str(actions_total[actions[agent_name]]).split()[0], red_act))
                # Perform action on the environment
                if self.messages:
                    messages = {
                        f'blue_agent_{i}': msg[i]
                        for i in range(5)
                    }
                    new_observations, reward, termination, truncation, _ = self.env.step(actions, messages=messages)
                else:
                    new_observations, reward, termination, truncation, _ = self.env.step(actions)
                # Append the rewards and termination for each agent
                done = []
                for i in range(5):
                    agent_name = f'blue_agent_{i}'
                    done.append(termination[agent_name] or truncation[agent_name])
                observations = new_observations
                if all(done):
                    break
                r.append(mean(reward.values()))  # Add rewards
            self.partial_rewards += sum(r)
            print(f"Final reward of the episode: {sum(r)}, length {self.count} - AVG: {self.partial_rewards / (eps + 1)}")
            # Add to partial rewards
            self.total_rewards.append(sum(r))
        save_statistics(self.total_rewards, self.total_rewards)
        for i in range(self.n_agents):
            agent_name = f'blue_agent_{i}'
            csv_filename = f"{agent_name}.csv"
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['Net', 'proc', 'Acts', 'Red_acts'])
                # Write data rows
                for data in self.agent_dict[agent_name]:
                    writer.writerow(data)

