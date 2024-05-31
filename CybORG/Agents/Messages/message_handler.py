import numpy as np

class MessageHandler:

    # Create 8-bit messages to send between agents
    def create_binary_message_full_bits_agent_4(self,malicious_process, malicious_network):
        message_size = 8 # 8-bits messages
        message = [0] * message_size
        process_bit = 0
        network_bit = 0
        process_count = 1
        network_count = 5
        for process in malicious_process:
            if any(process):
                process_bit = 1
                message[process_count] = 1
            process_count += 1
        for network in malicious_network:
            if any(network):
                network_bit = 1
                message[network_count] = 1
            network_count += 1
        message[0] = process_bit
        message[4] = network_bit
        return message
    
    def create_binary_message_two_bits(self,malicious_process, malicious_network):
        message_size = 8 # 8-bits messages
        message = [0] * message_size
        process_bit = 0
        network_bit = 0
        for process in malicious_process:
            if any(process):
                process_bit = 1
        for network in malicious_network:
            if any(network):
                network_bit = 1
        message[0] = process_bit
        message[1] = network_bit
        return np.array(message)
    
    # Return the number of malicious processes and networks identified
    def create_binary_message_full_bits(self, malicious_process,malicious_network):
        message_size = 8 # 8-bits messages
        message = [0] * message_size
        total_network = np.sum(malicious_network)
        total_processes = np.sum(malicious_process)
        binary_network = format(total_network, '04b')
        binary_processes = format(total_processes, '04b')
        binary_message = binary_processes + binary_network
        for i, bit in enumerate(binary_message):
            message[i] = int(bit)
        return message
    

    def action_messages(self, action_number):
        binary_str = bin(action_number)[2:]
        padded_binary_str = binary_str.zfill(8)
        # Convert the string to a list of integers
        binary_array = [int(bit) for bit in padded_binary_str]
    
        return np.array(binary_array)
    
    # Function to extract information for each subnet
    def extract_subnet_info(self, observation_vector, number):
        total_subnets = 1
        # Agent 4 takes care of more subnets
        if number == 4:
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
        # Iterate through each subnet information dictionary
        for subnet in subnet_info:
            # Append the 'malicious_network_event_detected' array to the malicious_network list
            malicious_network.append(subnet['malicious_network_event_detected'])
            malicious_process.append(subnet['malicious_process_event_detected'])
        #if number == 4:
            #return self.create_binary_message_full_bits_agent_4(malicious_process, malicious_network)
        #return self.create_binary_message_full_bits(malicious_process, malicious_network)
        return self.create_binary_message_two_bits(malicious_process, malicious_network)
