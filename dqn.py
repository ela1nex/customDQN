import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# environment
env = gym.make("CartPole-v1", render_mode="human") # creates the env

# action selection
def select_action(state, epsilon): # selects an action (random or not based on epsilon)
    return env.action_space.sample() # currently always random

# training loop
rewards = [] # rewards for each episode
steps = 0 # number of training(?) steps taken
for episode in range(5): # runs five episodes
    state = env.reset() # resets environment
    episode_reward = 0 # sets current reward to 0

    terminated = False 
    truncated = False

    while not terminated and not truncated:
        action = select_action(state, 1) # picks action based on current state and epsilon
        next, reward, terminated, truncated, info = env.step(action) # gets the feedback from the environment

        state = next # updates current state
        episode_reward += reward # adds step reward to episode reward

    rewards.append(episode_reward) # adds episode reward to rewards list
