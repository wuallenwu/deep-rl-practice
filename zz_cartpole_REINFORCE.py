#toy example of reinforce on gymnasium cartpole

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
env = gym.make('CartPole-v1')

class policy_net(nn.Module):
  def __init__(self, obs_dims : int, act_dims : int):
    super().__init__()
    self.fc1 = nn.Linear(obs_dims, 32)
    self.fc2 = nn.Linear(32, act_dims)
  
  def forward(self, state):
    state = torch.relu(self.fc1(state))
    # state = (state - state.mean()) / (state.std() + 1e-10)
    state = self.fc2(state)
    # state = torch.nan_to_num(state, nan = 0.5)

    # print(state)
    output = torch.softmax(state, dim = -1)
    return output

class REINFORCE:
  def __init__(self, obs_dims : int, act_dims : int):
    #hyperparameters
    self.alpha = 1e-3
    self.gamma = 0.99
    self.eps = 1e-9

    self.net = policy_net(obs_dims, act_dims)
    self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.alpha)
    self.probs = []
    self.rewards = []
  
  def sample(self, state):
    # probs = self.net(torch.from_numpy(state))
    # # print(probs)
    # action = torch.multinomial(probs, 1).item()
    # log_prob = torch.log(probs[:, action])
    # # print(log_prob)
    # self.probs.append(log_prob)
    # # print(action)
    # return action
    state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
    probs = self.net(state)
    # print(probs)
    action = torch.multinomial(probs, 1).item()
    log_prob = torch.log(probs[:, action])
    print(log_prob)
    # breakpoint()
    self.probs.append(log_prob)
    return action
  
  def update(self):
    #calculate rewards at each step
    # running_gamma = 0
    # disc_rewards = []
    # for r in self.rewards[::-1]:
    #     running_gamma = r + self.gamma * running_gamma
    #     disc_rewards.insert(0, running_gamma)

    # #policy gradient 
    # disc_rewards = torch.tensor(disc_rewards)
    # disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + self.eps)
    # policy_grad = []
    # for log_prob, r in zip(self.probs, disc_rewards):
    #   policy_grad.append(-log_prob * r)
    # self.optimizer.zero_grad()
    # policy_grad = torch.stack(policy_grad).sum()
    # policy_grad.backward()
    # self.optimizer.step()
    # self.probs = []
    # self.rewards = []

    print(self.probs)
    log_probs = torch.stack(self.probs)
    loss = -torch.mean(log_probs) * (sum(self.rewards) - 15)
    print("HIHIHIHIHI", loss)
    breakpoint()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.probs = []
    self.rewards = []

#hyperparameters
obs_dims = env.observation_space.shape[0]
act_dims = env.action_space.n
total_num_episodes = 10000
agent = REINFORCE(obs_dims, act_dims)
episode_rewards = []

for episode in range(total_num_episodes):
  obs, _ = env.reset(seed = 0)
  #run a single episode
  done = False
  score = 0
  while not done:
    action = agent.sample(obs)
    state, reward, done, _, _ = env.step(action)
    score += reward
    agent.rewards.append(reward)
    # done = x_out or angle_out or state[2]
    obs = state

  episode_rewards.append(score)
  # average_reward = []
  avg = 0
  if episode % 100 == 0:
    print(agent.alpha)
    if episode == 0: 
      # average_reward.append(score)
      print("Episode:", episode, "Average Reward:", score)
    else: 
      avg = np.mean(episode_rewards[episode - 100 : episode])
      # average_reward.append(avg)
      # print(average_reward)
      print("Episode:", episode, "Average Reward:", avg)
  agent.update()
  agent.alpha += episode * 1e-9

average_reward = [np.mean(episode_rewards[i:i+50]) for i in range(0,len(episode_rewards),50)]

#display learning over episodes
print(average_reward)
xs = [x for x in range(len(average_reward))]
plt.plot(xs, average_reward)
plt.show()
