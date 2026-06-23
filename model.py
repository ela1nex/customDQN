from torch import nn
import torch

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

# dynamic
class Dynamic(nn.Module):
    def __init__(self, input_dimensions):
        super(Dynamic, self).__init__()
        self.nn = nn.ModuleList()
        
        self.nn.append(nn.Linear(input_dimensions + 1, 256))  # state + action -> 256, wider since harder task than DQN
        self.nn.append(nn.Linear(256, 256))
        self.nn.append(nn.Linear(256, input_dimensions + 2))  # output: next_state + reward + done
    
    def forward(self, x):
        for layer in self.nn[:-1]:
            x = torch.relu(layer(x))
        return self.nn[-1](x)