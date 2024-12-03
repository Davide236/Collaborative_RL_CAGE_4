import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, lr, eps, fc):
        super(ActorCritic, self).__init__()

        # Actor Network: This network outputs action probabilities from the input state
        self.actor = nn.Sequential(
            nn.Linear(state_dim, fc),         # First hidden layer
            nn.ReLU(),                        # ReLU activation
            nn.Linear(fc, fc),                # Second hidden layer
            nn.ReLU(),                        # ReLU activation
            nn.Linear(fc, action_dim),        # Output layer (action probabilities)
            nn.Softmax(dim=-1)                # Softmax to convert outputs to probabilities
        )

        # Critic Network: This network estimates the value of the given state
        self.critic = nn.Sequential(
            nn.Linear(state_dim, fc),         # First hidden layer
            nn.ReLU(),                        # ReLU activation
            nn.Linear(fc, fc),                # Second hidden layer
            nn.ReLU(),                        # ReLU activation
            nn.Linear(fc, 1)                  # Output layer (state value)
        )

        # Optimizers for both the actor and critic networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=eps)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=eps)

    def action_selection(self, state):
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
                    We also compute the 'state value' of the state by querying the
                    critic network.
        """
        # Apply action mask to the action probabilities
        action_probs = self.actor(state)
        distribution = Categorical(action_probs)
        action = distribution.sample() # Exploration phase
        action_logprob = distribution.log_prob(action) # Compute log probability of action
        state_value = self.critic(state) # Compute the state value
        return action.detach(), action_logprob.detach(), state_value.detach() # Detach extra elements