import gymnasium as gym
import cv2
import torch

from agent import *

agent = Agent()

agent.critic.load_state_dict(torch.load("critic.pth"))
agent.critic.eval()
agent.dynamic.load_state_dict(torch.load("dynamic.pth"))
agent.dynamic.eval()

# testing loop
render_env = gym.make("CartPole-v1", render_mode="rgb_array")
state, info = render_env.reset()

terminated = False
truncated = False

episode_reward = 0
frames = []

while not terminated and not truncated:
    # both use dynamic model for training so the difference isnt huge but its still noticable
    # action = agent.select_action(state, 0, render_env) # does not use dynamic model for testing
    action = agent.select_planned_action(state) # uses dynamic model for testing
    
    next_state, reward, terminated, truncated, info = render_env.step(action)
    frames.append(render_env.render())

    state = next_state
    episode_reward += reward

render_env.close()

height, width, layers = frames[0].shape
video_size = (width, height)
fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('customdqn_cartpole_dynamic.mp4', fourcc, fps, video_size)

print(len(frames))

for frame in frames:
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(bgr_frame)

video.release()