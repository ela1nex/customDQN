# hyper params
learning_rate = 0.001 # learning rate of optimizer
gamma = 0.995 # discount factor
epsilon = 0.8 # starting epsilon value (random action chance)
epsilon_min = 0.01 # ending epsilon value
batch_size = 64 # number of transitions sampled from replay buffer
memory_size = 10000 # number of transitions stored for sampling
training_steps = 100000 # steps to train
tau = 0.005 # update rate of target network
verbose = 1 # training info printing (0 is no info, 1 is info)
log_interval = 1000 # interval of episodes to print info

imagination_rollout_length = 2 # number of rollouts done for the imaginated buffer

# do not change unless changing environment
input_dimensions = 4 # based off the shape of the environment (4 for cart pole)
output_dimensions = 2 # based off the number of action spaces of the environment (2 for cart pole)