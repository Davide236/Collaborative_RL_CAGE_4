import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch

class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def std(self):
        return np.sqrt(self.var)
    
class RDN_Network(nn.Module):
    def __init__(self, state_dim):
        super(RDN_Network, self).__init__()   
        fc = 256
        self.predictor_network = nn.Sequential(
            nn.Linear(state_dim, fc),
            nn.ReLU(),
            nn.Linear(fc, fc),
            nn.ReLU(),
            nn.Linear(fc, 1)
        ).float()

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, fc),
            nn.ReLU(),
            nn.Linear(fc, fc),
            nn.ReLU(),
            nn.Linear(fc, 1)
        ).float()
        self.eps = 0.001
        self.k = 30
        self.L = 5
        self.episodic_memory = []
        self.rnd_optimizer = optim.Adam(self.predictor_network.parameters(), lr=1e-4)
        self.scheduler = ExponentialLR(self.rnd_optimizer, 0.999)
        self.running_mean_std = RunningMeanStd()
        self.dm = 0.0  # Running average of squared distances of the k-th nearest neighbors

    def compute_similarity(self, state):
        knn = self.k
        # Check if memory is empty
        if len(self.episodic_memory) == 0:
            return 1
        elif len(self.episodic_memory) < self.k:
            knn = len(self.episodic_memory)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        memory_tensors = torch.FloatTensor(np.array(self.episodic_memory))
        # Pairwise distance between the current state and states in memory
        distances = torch.cdist(state_tensor, memory_tensors).squeeze(0)
        # Find the K smallest distances (K nearest neighbors)
        knn_distances, _ = torch.topk(distances, knn, largest=False)
        knn_distances = knn_distances.cpu().numpy()

        # Update running average of squared distances of the k-th nearest neighbors
        self.dm = 0.99 * self.dm + 0.01 * np.mean(knn_distances ** 2)

        kernel_sum = np.sum(self.eps / (knn_distances ** 2 / self.dm + self.eps))
        similarity = np.sqrt(kernel_sum + self.eps)
        return similarity
    

    def compute_rnd_alpha(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        target_output = self.target_network(state_tensor).detach()
        predictor_output = self.predictor_network(state_tensor)
        prediction_error = torch.mean((target_output - predictor_output) ** 2, dim=1).item()
        
        # Normalize the prediction error
        self.running_mean_std.update([prediction_error])
        normalized_prediction_error = prediction_error / self.running_mean_std.std()
        
        return normalized_prediction_error
    

    def compute_intrinsic_reward(self, state):
        similarity = self.compute_similarity(state)
        episodic_reward = 1.0 / similarity
        alpha = self.compute_rnd_alpha(state)
        print(episodic_reward, alpha)
        intrinsic_reward = episodic_reward * np.clip(alpha, 1, self.L)
        self.episodic_memory.append(state)
        return intrinsic_reward
    

    def update_predictor(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        target_output = self.target_network(state_tensor).detach()
        predictor_output = self.predictor_network(state_tensor)
        loss = torch.mean((target_output - predictor_output) ** 2)
        
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
    
    def anneal_lr(self):
        self.scheduler.step()
    
    def reset_memory(self):
        self.episodic_memory.clear()
    