import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch

# Class to compute running mean and standard deviation for normalization
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    # Update running mean and standard deviation based on new data
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    # Update running statistics (mean, variance) based on moments
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        # Update mean and variance using the formula for running averages
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    # Return the standard deviation (square root of variance)
    def std(self):
        return np.sqrt(self.var)

# Class defining the RDN (Random Network Distillation) network
class RDN_Network(nn.Module):
    def __init__(self, state_dim):
        super(RDN_Network, self).__init__()   
        fc = 256  # Hidden layer size

        # Predictor Network: Network used for making predictions of the target network's output
        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, fc),
            nn.ReLU(),
            nn.Linear(fc, fc),
            nn.ReLU(),
            nn.Linear(fc, 1)
        ).float()

        # Target Network: Target used to compute the error for training the predictor
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, fc),
            nn.ReLU(),
            nn.Linear(fc, fc),
            nn.ReLU(),
            nn.Linear(fc, 1)
        ).float()

        # Hyperparameters for RND
        self.eps = 0.001  # Small epsilon value to avoid division by zero
        self.k = 30  # Number of nearest neighbors to consider
        self.L = 5  # Maximum possible intrinsic reward scaling factor

        # Memory to store states (episodic memory)
        self.episodic_memory = []

        # Optimizer for the predictor network
        self.rnd_optimizer = optim.Adam(self.predictor_network.parameters(), lr=1e-4)

        # Learning rate scheduler (exponential decay)
        self.scheduler = ExponentialLR(self.rnd_optimizer, 0.999)

        # Running statistics to normalize prediction errors
        self.running_mean_std = RunningMeanStd()

        # Running average of squared distances for the k-th nearest neighbors
        self.dm = 0.001

    # Compute the similarity of the current state with the episodic memory
    def compute_similarity(self, state):
        knn = self.k
        # If memory is empty, return maximum similarity (1)
        if len(self.episodic_memory) == 0:
            return 1
        # If memory size is less than k, use the full memory
        elif len(self.episodic_memory) < self.k:
            knn = len(self.episodic_memory)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        memory_tensors = torch.FloatTensor(np.array(self.episodic_memory))  # Convert memory to tensor

        # Compute pairwise distances between the current state and states in memory
        distances = torch.cdist(state_tensor, memory_tensors).squeeze(0)

        # Get the k smallest distances (k nearest neighbors)
        knn_distances, _ = torch.topk(distances, knn, largest=False)
        knn_distances = knn_distances.cpu().numpy()  # Convert distances to numpy for further processing

        # Update running average of squared distances of the k nearest neighbors
        self.dm = 0.99 * self.dm + 0.01 * np.mean(knn_distances ** 2)

        # Compute similarity using kernel sum
        kernel_sum = np.sum(self.eps / (knn_distances ** 2 / self.dm + self.eps))
        similarity = np.sqrt(kernel_sum + self.eps)

        return similarity

    # Compute the normalized prediction error (alpha) for the given state
    def compute_rnd_alpha(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor

        # Get output of target network (no gradient computation)
        target_output = self.target_network(state_tensor).detach()

        # Get output of predictor network
        predictor_output = self.predictor_network(state_tensor)

        # Compute the prediction error
        prediction_error = torch.mean((target_output - predictor_output) ** 2, dim=1).item()

        # Normalize the prediction error using running mean and std
        self.running_mean_std.update([prediction_error])
        normalized_prediction_error = prediction_error / self.running_mean_std.std()

        return normalized_prediction_error

    # Compute the intrinsic reward for the given state based on similarity and prediction error
    def compute_intrinsic_reward(self, state):
        similarity = self.compute_similarity(state)  # Compute similarity with memory
        episodic_reward = 1.0 / similarity  # Reward inversely proportional to similarity

        # Compute the prediction error and alpha (adjustment factor)
        alpha = self.compute_rnd_alpha(state)

        # Compute intrinsic reward by scaling episodic reward by alpha (clipped to range [1, L])
        intrinsic_reward = episodic_reward * np.clip(alpha, 1, self.L)

        # Add current state to episodic memory
        self.episodic_memory.append(state)

        return intrinsic_reward

    # Update the predictor network using the current state
    def update_predictor(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor

        # Get output of target network (no gradient computation)
        target_output = self.target_network(state_tensor).detach()

        # Get output of predictor network
        predictor_output = self.predictor_network(state_tensor)

        # Compute loss between target and predictor output
        loss = torch.mean((target_output - predictor_output) ** 2)

        # Backpropagate the loss and update the predictor network
        self.rnd_optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation
        self.rnd_optimizer.step()  # Optimizer step

    # Anneal the learning rate using the scheduler
    def anneal_lr(self):
        self.scheduler.step()

    # Reset episodic memory
    def reset_memory(self):
        self.episodic_memory.clear()  # Clear the memory of states
