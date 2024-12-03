import numpy as np

class MessageHandler:
    def __init__(self, message_type, number):
        """
        Args:
            message_type (str): Type of message to be created ('2_bits', 'action', etc.).
            number (int): The number of the agent (affects the subnet handling).

        Returns:
            None

        Explanation:
            Initializes the MessageHandler with the type of message to be created and the agent's number.
        """
        super(MessageHandler, self).__init__()
        self.message_type = message_type  # Set the type of message to be created
        self.agent_number = number  # Set the agent number to determine its subnet handling

    def create_binary_message_full_bits_agent_4(self, malicious_process, malicious_network):
        """
        Args:
            malicious_process (list): A list of malicious process detections for each subnet.
            malicious_network (list): A list of malicious network detections for each subnet.

        Returns:
            np.array: A binary message encoded as an 8-bit numpy array.

        Explanation:
            Creates an 8-bit message based on the malicious processes and networks. The first bit represents 
            any detected malicious process, the fifth bit represents any detected malicious network, and 
            the remaining bits represent each process and network detection.
        """
        message_size = 8  # 8-bits messages
        message = [0] * message_size  # Initialize the message with all 0s
        process_bit = 0
        network_bit = 0
        process_count = 1
        network_count = 5

        # Set the process bits based on detected malicious processes
        for process in malicious_process:
            if any(process):
                process_bit = 1
                message[process_count] = 1
            process_count += 1

        # Set the network bits based on detected malicious networks
        for network in malicious_network:
            if any(network):
                network_bit = 1
                message[network_count] = 1
            network_count += 1

        # Set the first and fifth bits for overall malicious process and network detection
        message[0] = process_bit
        message[4] = network_bit

        return np.array(message)

    def create_binary_message_two_bits(self, malicious_process, malicious_network):
        """
        Args:
            malicious_process (list): A list of malicious process detections for each subnet.
            malicious_network (list): A list of malicious network detections for each subnet.

        Returns:
            np.array: A binary message encoded as an 8-bit numpy array.

        Explanation:
            Creates an 8-bit message where the first bit represents any malicious process detection,
            and the second bit represents any malicious network detection. This simplifies the message 
            by using only two bits.
        """
        message_size = 8  # 8-bits messages
        message = [0] * message_size  # Initialize the message with all 0s
        process_bit = 0
        network_bit = 0

        # Check for any malicious process and set the first bit
        for process in malicious_process:
            if any(process):
                process_bit = 1

        # Check for any malicious network and set the second bit
        for network in malicious_network:
            if any(network):
                network_bit = 1

        message[0] = process_bit  # Set the first bit for process detection
        message[1] = network_bit  # Set the second bit for network detection

        return np.array(message)

    def create_binary_message_full_bits(self, malicious_process, malicious_network):
        """
        Args:
            malicious_process (list): A list of malicious process detections for each subnet.
            malicious_network (list): A list of malicious network detections for each subnet.

        Returns:
            np.array: A binary message encoded as an 8-bit numpy array.

        Explanation:
            Creates an 8-bit binary message. The first 4 bits represent the total number of malicious 
            processes detected (in binary), and the last 4 bits represent the total number of malicious 
            networks detected (in binary).
        """
        message_size = 8  # 8-bits messages
        message = [0] * message_size  # Initialize the message with all 0s
        total_network = np.sum(malicious_network)  # Sum of malicious network detections
        total_processes = np.sum(malicious_process)  # Sum of malicious process detections

        # Convert the totals to 4-bit binary strings
        binary_network = format(int(total_network), '04b')
        binary_processes = format(int(total_processes), '04b')

        # Combine the binary strings and store in the message
        binary_message = binary_processes + binary_network
        for i, bit in enumerate(binary_message):
            message[i] = int(bit)

        return np.array(message)

    def action_messages(self, action_number):
        """
        Args:
            action_number (int): The action number to be encoded in binary.

        Returns:
            np.array: An 8-bit binary message representing the action number.

        Explanation:
            Converts the given action number into an 8-bit binary string, padded with leading zeros, 
            and returns it as a numpy array.
        """
        binary_str = bin(action_number)[2:]  # Convert action number to binary string
        padded_binary_str = binary_str.zfill(8)  # Pad the binary string to ensure it is 8 bits
        binary_array = [int(bit) for bit in padded_binary_str]  # Convert the binary string to a list of integers

        return np.array(binary_array)

    def extract_subnet_info(self, observation_vector):
        """
        Args:
            observation_vector (list): A list of observations representing the state of the network.

        Returns:
            np.array: A binary message that encodes the subnet information.

        Explanation:
            Extracts the information about subnets from the given observation vector and returns it 
            as a binary message. The number of subnets handled depends on the agent number (Agent 4 handles more subnets).
        """
        total_subnets = 1  # Default total subnets
        if self.agent_number == 4:
            total_subnets = 3  # Agent 4 handles 3 subnets
        S = 9  # Number of subnets
        H = 16  # Maximum number of hosts per subnet
        subnets_length = 3 * S + 2 * H  # Length of the subnet information
        subnet_info = []

        # Iterate through the subnets to extract information
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

        # Collect malicious network and process events from the subnet information
        malicious_network = []
        malicious_process = []
        for subnet in subnet_info:
            malicious_network.append(subnet['malicious_network_event_detected'])
            malicious_process.append(subnet['malicious_process_event_detected'])

        # Depending on message type, return the appropriate message
        if self.message_type == '2_bits':
            return self.create_binary_message_two_bits(malicious_process, malicious_network)
        
        # Return the message based on agent's capabilities
        if self.agent_number == 4:
            return self.create_binary_message_full_bits_agent_4(malicious_process, malicious_network)

        return self.create_binary_message_full_bits(malicious_process, malicious_network)

    def prepare_message(self, state, action_number):
        """
        Args:
            state (list): The current state (observation vector) of the agent.
            action_number (int): The action to be taken by the agent.

        Returns:
            np.array: The prepared message to be sent by the agent.

        Explanation:
            Prepares a message based on the message type ('action' or 'subnet info') and the state or action number.
        """
        if self.message_type == 'action':
            return self.action_messages(action_number)  # Create message for action
        return self.extract_subnet_info(state)  # Create message with subnet information
