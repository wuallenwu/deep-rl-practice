import gymnasium as gym
import numpy as np

seed = 0
env = gym.make("MountainCar-v0")
np.random.seed(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

# hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate of exploration probability
total_episodes = 2000
obs_dims = env.observation_space.shape[0]
os_bin_size = 20
display_interval = 100

# initialize state bins and Q table
os_bins = [os_bin_size] * obs_dims
Q = np.random.uniform(low=-1, high=1, size=os_bins + [env.action_space.n])


def discretize_state(state):
    # Normalize state to be between 0 and 1
    state_adj = (state - env.observation_space.low) / (
        env.observation_space.high - env.observation_space.low
    )
    # Convert to discrete bins
    state_discrete = np.floor(state_adj * os_bin_size).astype(int)
    state_discrete = np.clip(state_discrete, 0, os_bin_size - 1)  # Ensure within bounds
    return tuple(state_discrete)


for episode in range(total_episodes):
    obs, _ = env.reset(seed=seed)

    obs = discretize_state(obs)
    done = False
    step = 0
    total_reward = 0  # Track total reward for the episode

    while not done:
        step += 1
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: select a random action
        else:
            action = np.argmax(Q[obs])  # Exploit: select the action with max Q-value

        cur_obs, reward, terminated, truncated, _ = env.step(action)
        cur_obs = discretize_state(cur_obs)

        # Update Q-table
        Q[obs][action] = Q[obs][action] + alpha * (
            reward + gamma * np.max(Q[cur_obs]) - Q[obs][action]
        )

        obs = cur_obs
        total_reward += reward

        if terminated:
            print(f"Episode {episode} won after {step} steps")

        done = terminated or truncated

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()
