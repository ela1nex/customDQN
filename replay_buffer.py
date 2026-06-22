from collections import namedtuple, deque
import random
import torch
import numpy as np

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