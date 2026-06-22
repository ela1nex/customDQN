import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
from collections import deque, namedtuple
from replay_buffer import *

from configs import *

# dqn
class DQN(nn.Module): # subclasses nn.Module
    def __init__(self, input_dimensions, output_dimensions, layers): # initializes the nn w/ the layers and input/output dimensions given
        super(DQN, self).__init__()
        self.nn = nn.ModuleList()
        self.nn.append(nn.Linear(input_dimensions, 128)) # input dimension -> 128
        for layer in range(layers-2):
            self.nn.append(nn.Linear(128, 128)) # layers in between
        self.nn.append(nn.Linear(128, output_dimensions)) # 128 -> output dimension

    def forward(self, x):
        for layer in range(len(self.nn)-1):
            x = torch.relu(self.nn[layer](x)) # passes input through all layers except last with relu activation function
        x = self.nn[-1](x) # returns output 
        return x

# q-network initialization
critic = DQN(input_dimensions, output_dimensions, 3) # critic network
target = DQN(input_dimensions, output_dimensions, 3) # target network
dynamic = DQN(5, 6, 3) # dynamic network
target.load_state_dict(critic.state_dict()) # copies weights from critic network
target.eval() # switches from training mode to evaluation mode

critic_optimizer = optim.AdamW(critic.parameters(), lr=learning_rate, amsgrad=True) # optimizer for critic based off defined learning rate
dynamic_optimizer = optim.AdamW(dynamic.parameters(), lr=learning_rate, amsgrad=True) # optimizer for dynamic
memory = Memory(memory_size) # the memory of the optimizer with defined max length

# action selection
def select_action(state, epsilon, env): # selects an action (random or not based on epsilon)
    if random.random() < epsilon: # random for epsilon*100 percent of the time
        return env.action_space.sample() # explores
    else:
        state = torch.FloatTensor(state).unsqueeze(0) # adds dimension of size 1 at position 0
        q_values = critic(state) # gets the q values for the current state from the critic
        return torch.argmax(q_values).item() # exploits

def optimize_model():
    if len(memory) < batch_size: # if the memory is smaller than batch size then it wont optimize
        return
    
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size) # randomly samples a batch of batch_size from the memory

    # TODO dynamic loss -- take state + action vector as input and output reward, next_state, done
    state_action_batch = torch.cat((state_batch, action_batch.float()), dim=1) # concatenates state and action batches
    dynamic_output = dynamic(state_action_batch) # get the predicted reward, next state, and done from dynamic network
    dynamic_reward = dynamic_output[:, 0] # extract predicted reward
    dynamic_next_state = dynamic_output[:, 1:1+input_dimensions] # extract predicted next state
    dynamic_done = dynamic_output[:, -1] # extract predicted done
    
    dynamic_loss = nn.MSELoss()(dynamic_reward, reward_batch) + nn.MSELoss()(dynamic_next_state, next_state_batch) + nn.BCEWithLogitsLoss()(dynamic_done, done_batch) # calculates loss for dynamic network
    global last_dynamic_loss
    last_dynamic_loss = dynamic_loss.item()

    dynamic_optimizer.zero_grad()
    dynamic_loss.backward()
    dynamic_optimizer.step()

    q_values = critic(state_batch).gather(1, action_batch).squeeze() # predicts q-values for all actions and extracts value of action actually taken

    with torch.no_grad(): # does not remember operations   
        max_next_q_values = target(next_state_batch).max(1)[0] # outputs best possible q-value from next state
        target_q_values = reward_batch + gamma * max_next_q_values * (1-done_batch) # calculates immediate reward and future estimated reward
    
    critic_loss = nn.MSELoss()(q_values, target_q_values) # calculates distance from predicated q-values to target q-value
    global last_critic_loss
    last_critic_loss = critic_loss.item()

    critic_optimizer.zero_grad() # clears old gradients
    critic_loss.backward() # computes gradients of loss w.r.t. model parameters
    critic_optimizer.step() # updatse network weights 

def average(data, window=log_interval):
    window = min(len(data), window)
    return sum(data[-window:])/window