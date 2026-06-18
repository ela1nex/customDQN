import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import cv2
from collections import deque, namedtuple

# environment
env = gym.make("CartPole-v1") # creates the env

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

# memory
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
class Memory(object): 
    def __init__(self, memory_size): # initializes the memory as a deque with a given max size 
        self.memory = deque([], maxlen=memory_size)
    
    def push(self, *args): # pushes a transition into the memory
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size): # randomly samples a batch from the memory
        batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # turn everything into tensors
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32) 
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch) 
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.FloatTensor(done_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self): # returns the size of the memory
        return len(self.memory)

# hyper params
learning_rate = 0.001 # learning rate of optimizer
gamma = 0.995 # discount factor
epsilon = 0.8 # starting epsilon value (random action chance)
epsilon_min = 0.01 # ending epsilon value
epsilon_decay = 0.995 # epsilon decay rate
batch_size = 64 # number of transitions sampled from replay buffer
memory_size = 10000 # number of transitions stored for sampling
episodes = 500 # episodes to train
tau = 0.005 # update rate of target network
verbose = 1 # training info printing
log_interval = 10 # interval of training steps to print info

# q-network initialization
input_dimensions = env.observation_space.shape[0] # based off the shape of the environment (4 for cart pole)
output_dimensions = env.action_space.n # based off the number of action spaces of the environment (2 for cart pole)
critic = DQN(input_dimensions, output_dimensions, 3) # critic network
target = DQN(input_dimensions, output_dimensions, 3) # target network
target.load_state_dict(critic.state_dict()) # copies weights from critic network
target.eval() # switches from training mode to evaluation mode

optimizer = optim.AdamW(critic.parameters(), lr = learning_rate, amsgrad=True) # optimizer for critic based off defined learning rate
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

    q_values = critic(state_batch).gather(1, action_batch).squeeze() # predicts q-values for all actions and extracts value of action actually taken

    with torch.no_grad(): # does not remember operations   
        max_next_q_values = target(next_state_batch).max(1)[0] # outputs best possible q-value from next state
        target_q_values = reward_batch + gamma * max_next_q_values * (1-done_batch) # calculates immediate reward and future estimated reward
    
    critic_loss = nn.MSELoss()(q_values, target_q_values) # calculates distance from predicated q-values to target q-value
    global last_loss
    last_loss = critic_loss.item()

    optimizer.zero_grad() # clears old gradients
    critic_loss.backward() # computes gradients of loss w.r.t. model parameters
    optimizer.step() # updatse network weights 

def average(data, window=log_interval):
    window = min(len(data), window)
    return sum(data[-window:])/window

# training loop
rewards = [] # rewards for each episode
lengths = [] # lengths for each episode
steps = 0 # number of training steps taken
for episode in range(episodes): # runs given number of episodes
    state, info = env.reset() # resets environment
    episode_reward = 0 # sets current reward to 0
    episode_length = steps

    terminated = False 
    truncated = False

    while not terminated and not truncated:
        action = select_action(state, epsilon, env) # picks action based on current state and epsilon
        next_state, reward, terminated, truncated, info = env.step(action) # gets the feedback from the environment

        memory.push(state, action, reward, next_state, terminated or truncated) # add step to memory

        state = next_state # updates current state
        episode_reward += reward # adds step reward to episode reward

        optimize_model()

        # uses tau to soft update the target network weights
        target_state_dict = target.state_dict()
        critic_state_dict = critic.state_dict()
        for key in critic_state_dict:
            target_state_dict[key] = critic_state_dict[key]*tau + target_state_dict[key]*(1-tau)
        target.load_state_dict(target_state_dict)

        steps += 1
    
    if verbose == 1 and episode%log_interval == 0 and episode != 0:
            print(f"------------- \nstep: {steps} \nepisode: {episode} \navg length: {average(lengths)} \navg reward: {average(rewards)} \nloss: {last_loss} \nepsilon: {epsilon}")

    epsilon = max(epsilon_min, epsilon_decay * epsilon) # decays epsilon

    episode_length = steps - episode_length # calculates length of current episode
    lengths.append(episode_length) # adds episode length to lengths list
    rewards.append(episode_reward) # adds episode reward to rewards list

torch.save(critic, "model.pth")

# testing loop
render_env = gym.make("CartPole-v1", render_mode="rgb_array")
state, info = render_env.reset()

terminated = False
truncated = False

episode_reward = 0
frames = []

while not terminated and not truncated:
    action = select_action(state, 0, render_env)
    next_state, reward, terminated, truncated, info = render_env.step(action)
    frames.append(render_env.render())

    state = next_state
    episode_reward += reward

render_env.close()

height, width, layers = frames[0].shape
video_size = (width, height)
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('customdqn_cartpole.mp4', fourcc, fps, video_size)

print(len(frames))

for frame in frames:
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(bgr_frame)

video.release()