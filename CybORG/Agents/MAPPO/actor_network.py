import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, lr, eps, fc):
        super(ActorNetwork, self).__init__()
        # Width of the network
        # Initialize actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc),
            nn.ReLU(),
            nn.Linear(fc,fc),
            nn.ReLU(),
            nn.Linear(fc, action_dim),
            nn.Softmax(dim=-1) # For probabilities
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=eps)

    def action_selection(self, state, action_mask):
        """
        Args: 
            state: The current observation state of the agent.

        Returns:
            action: an action sampled from the action distribution
            log probability: the logarithmic probability of the action (given the distribution)
                            state value: the value associated with the state

        Explanation: From the input state, the Actor network is queried
                    and returns an action probability distribution, from which
                    we sample an action and its associated log probability value.
        """
        # Apply action mask to the action probabilities
        #masked_action_probs = self.actor(state)
        masked_action_probs = torch.tensor(action_mask, dtype=torch.float) * self.actor(state)
        masked_action_probs /= masked_action_probs.sum()
        distribution = Categorical(masked_action_probs)
        action = distribution.sample() # Exploration phase
        action_logprob = distribution.log_prob(action) # Compute log probability of action
        # TODO: Compute state value in other action
        return action.detach(), action_logprob.detach()# Detach extra elements
    
    def forward(self, state):
        return self.actor(state)