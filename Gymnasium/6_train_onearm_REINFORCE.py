# REINFORCE for one arm touching block

from collections import OrderedDict
import random
import time
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym
import gym_kmanip
from gymnasium.wrappers import RecordVideo

total_num_episodes = int(5e4)  # Total number of episodes
training_period = total_num_episodes / 10
update_period = int(5e2)
obs_space_dims, action_space_dims = 23, 7 #23 observation space dims since we don't care about cube orientation

plt.rcParams["figure.figsize"] = (10, 5)
torch.set_printoptions(precision=8)

# choose your environment
ENV_NAME: str = "KManipSoloArm"
# ENV_NAME: str = "KManipSoloArmQPos"
# ENV_NAME: str = "KManipSoloArmVision"
# ENV_NAME: str = "KManipDualArm"
# ENV_NAME: str = "KManipDualArmVision"
# ENV_NAME: str = "KManipTorso"
# ENV_NAME: str = "KManipTorsoVision"
env = gym.make(ENV_NAME)
env = RecordVideo(env, video_folder="SoloArmAgent", name_prefix="training",
                  episode_trigger=lambda x: x % training_period == 0)
# env = RecordEpisodeStatistics(env)

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  
        hidden_space2 = 32  

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )
        return action_means, action_stddevs

class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 3e-4  # Learning rate for policy optimization
        self.gamma = 0.999  # Discount factor
        self.eps = 1e-6   # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        flattened = np.concatenate(list(state.values()))
        for f in flattened:
            if abs(f) > 1:
                print("invalid observation" + flattened)
                breakpoint()
        flattened = flattened[:23]
        state = torch.tensor(flattened, dtype=torch.float64)
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        # mean and standard deviation and sample an action
        distribs = [Normal(action_means[i] + self.eps, action_stddevs[i] + self.eps) for i in range(len(action_means))]
        actions = [(distrib.sample()) for distrib in distribs]

        #find log probability of taking the action
        log_probs = [distribs[i].log_prob(actions[i]) for i in range(len(distribs))]
        log_prob_sum = torch.sum(torch.stack(log_probs))
        self.probs.append(log_prob_sum)

        actions = torch.tensor([action.item() for action in actions], dtype = torch.float64)
        actions = actions.numpy()
        #minmax scale
        actions = actions / (max(abs(np.min(actions)), abs(np.max(actions))))
        #make sure all actions in range
        for a in actions:
            if abs(a) > 1:
                print("invalid action" + actions)
                breakpoint()

        keys = ["eer_pos", "eer_orn", "grip_r"]
        vals = [np.array(actions[:3:]), np.array(actions[3:6:]), np.array(actions[6])]

        actiondict = OrderedDict()
        for i in range (3):
            actiondict[keys[i]] = vals[i]
        return actiondict

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)
        deltas = torch.tensor(gs)
        loss = 0

        # minimize LOSS: -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas, strict = True):
            loss += log_prob * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()

        # DEBUG gradients
        # for name, param in self.net.named_parameters():
        # print('Grad Sum', torch.sum([torch.norm(param.grad) ])
        # gnorm = nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
        # print(gnorm)
        

        self.optimizer.step()
        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

rewards_over_seeds = []

for seed in [0, 3, 5]:  
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []
    touches = 0
    for episode in range(total_num_episodes + 1):
        start_time = time.time()
        # set random seed for consistent tests
        obs, _ = env.reset(seed=seed)
        done = False
        rewards = []
        signal = False

        while not done:
            action = agent.sample_action(obs)
            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = env.step(action)
            if reward > 50:
                touches += 1
                # print("touch signal sent")
                # signal = True
                # breakpoint()
            rewards.append(reward)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated 

        agent.update()
        # if(signal == True):
            # print(rewards)
            # breakpoint()
        reward_over_episodes.append(np.average(rewards))
        if episode == 0:
            print ("Episode 0", reward_over_episodes[0])
        if episode % update_period == 0 and episode != 0:
            print("Episode:", episode, "Average Reward:", np.average(reward_over_episodes[episode-update_period:episode]), "Total Touches over", int(update_period), "episodes:", touches)
            touches = 0
        logging.info(f"episode-{episode}", info["episode"]) 
    env.close()

#display learning over episodes
xs = [x for x in range(len(reward_over_episodes))]
plt.plot(xs, reward_over_episodes)
plt.show()