# hyper params
learning_rate = 0.001 # learning rate of optimizer
gamma = 0.995 # discount factor
epsilon = 0.8 # starting epsilon value (random action chance)
epsilon_min = 0.01 # ending epsilon value
epsilon_decay = 0.995 # epsilon decay rate
batch_size = 64 # number of transitions sampled from replay buffer
memory_size = 10000 # number of transitions stored for sampling
episodes = 5000 # episodes to train
tau = 0.005 # update rate of target network
verbose = 1 # training info printing
log_interval = 10 # interval of episodes to print info

input_dimensions = 4 # based off the shape of the environment (4 for cart pole)
output_dimensions = 2 # based off the number of action spaces of the environment (2 for cart pole)