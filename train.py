import gymnasium as gym
import torch

from configs import *
from agent import *

def average(data, window=log_interval):
    window = min(len(data), window)
    return sum(data[-window:])/window

# environment
env = gym.make("CartPole-v1") # creates the env

agent = Agent()

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
        action = agent.select_action(state, epsilon, env) # picks action based on current state and epsilon
        next_state, reward, terminated, truncated, info = env.step(action) # gets the feedback from the environment

        agent.memory.push(state, action, reward, next_state, terminated or truncated) # add step to memory

        state = next_state # updates current state
        episode_reward += reward # adds step reward to episode reward

        critic_loss, dynamic_loss = agent.optimize_model()
        last_critic_loss, last_dynamic_loss = critic_loss, dynamic_loss

        # uses tau to soft update the target network weights
        target_state_dict = agent.target.state_dict()
        critic_state_dict = agent.critic.state_dict()
        for key in critic_state_dict:
            target_state_dict[key] = critic_state_dict[key]*tau + target_state_dict[key]*(1-tau)
        agent.target.load_state_dict(target_state_dict)

        steps += 1
    
    if verbose == 1 and episode%log_interval == 0 and episode != 0:
            print(f"------------- \nstep: {steps} \nepisode: {episode} \navg length: {average(lengths)} \navg reward: {average(rewards)} \ncritic loss: {last_critic_loss} \ndynamic loss: {last_dynamic_loss} \nepsilon: {epsilon}")

    epsilon = max(epsilon_min, epsilon_decay * epsilon) # decays epsilon

    episode_length = steps - episode_length # calculates length of current episode
    lengths.append(episode_length) # adds episode length to lengths list
    rewards.append(episode_reward) # adds episode reward to rewards list

torch.save(agent.critic.state_dict(), "critic.pth")
torch.save(agent.dynamic.state_dict(), "dynamic.pth")