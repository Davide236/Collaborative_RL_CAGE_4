import collections
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_limit, n_agents, obs_space):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.n_agents = n_agents
        self.obs_space = obs_space

    def put(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        return torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents, self.obs_space), \
               torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents), \
               torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents), \
               torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, self.n_agents, self.obs_space), \
               torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1)

    def size(self):
        return len(self.buffer)