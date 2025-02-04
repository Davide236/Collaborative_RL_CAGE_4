# Reinforcement Learning for Cybersecurity Simulations

## Overview
This project is a Master's Thesis focused on evaluating various open-source reinforcement learning (RL) environments that simulate different cybersecurity scenarios. The evaluation was followed by the implementation of RL techniques to defend the infrastructure within the selected environment, **CybORG - CAGE 4**. The project also assessed the effectiveness of the learned policy under different conditions, including modifications to the reward schema and network topology.

This research was conducted in collaboration with **TNO** and **Rijksuniversiteit Groningen**.

- **CAGE 4 Page**: [CAGE 4 Repository](https://github.com/cage-challenge/cage-challenge-4)

## Training Different Models
The project provides implementations for training different RL algorithms on the **CybORG** environment. The available training methods include:

- **Independent PPO (IPPO)**
- **Multi-Agent PPO (MAPPO)**
- **Recurrent Independent PPO (R-IPPO)**
- **Recurrent Multi-Agent PPO (R-MAPPO)**
- **MADDPG**
- **QMIX**
- **Recurrent QMIX (R-QMIX)**

### Running the Training Script
To train an RL agent, use:
```sh
python train.py --Method <ALGORITHM> --Messages <True/False> --Load_last <True/False> --Load_best <True/False> --Rollout <N> --Episodes <N>
```

### Argument Descriptions
| Argument      | Description |
|--------------|-------------|
| `--Method`   | The RL algorithm to use (default: IPPO) |
| `--Messages` | Enable/disable messages during training (default: False) |
| `--Load_last` | Load the last saved model (default: False) |
| `--Load_best` | Load the best saved model (default: False) |
| `--Rollout`  | Number of episodes stored before training (default: 10) |
| `--Episodes` | Total number of training episodes (default: 4000) |

## Evaluating Trained Models
After training, you can evaluate the performance of the trained RL models using:
```sh
python evaluate.py --Method <ALGORITHM> --Messages <True/False> --Load_last <True/False> --Load_best <True/False>
```

## Computing the Agent Policy
After evaluation, the agent policy can be analyzed using the `compute_agent_policy.py` script located in the `statistics` folder. This script processes evaluation data to extract insights about the learned policy and how it adapts under different conditions.

## License
This project is open-source and intended for research and educational purposes.

---
For more details, refer to the **Thesis Report**, where the methodology, experimental setup, and findings are discussed in depth.

