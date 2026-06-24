import random
import torch
from torch import nn, optim

from replay_buffer import Transition, Memory
from model import DQN, Dynamic
from configs import *

class Agent():
    def __init__(self):
        # q-network initialization
        self.critic = DQN(input_dimensions, output_dimensions, 3) # critic network
        self.target = DQN(input_dimensions, output_dimensions, 3) # target network
        self.dynamic = Dynamic(input_dimensions) # dynamic network
        self.target.load_state_dict(self.critic.state_dict()) # copies weights from critic network
        self.target.eval() # switches from training mode to evaluation mode

        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=learning_rate, amsgrad=True) # optimizer for critic based off defined learning rate
        self.dynamic_optimizer = optim.AdamW(self.dynamic.parameters(), lr=learning_rate, amsgrad=True) # optimizer for dynamic
        self.memory = Memory(memory_size) # the memory of the optimizer with defined max length
        self.imagined_memory = Memory(memory_size) # memory for dynamic training
        
    # action selection
    def select_action(self, state, epsilon, env): # selects an action (random or not based on epsilon)
        if random.random() < epsilon: # random for epsilon*100 percent of the time
            return env.action_space.sample() # explores
        else:
            state = torch.FloatTensor(state).unsqueeze(0) # adds dimension of size 1 at position 0
            q_values = self.critic(state) # gets the q values for the current state from the critic
            return torch.argmax(q_values).item() # exploits

    def optimize_model(self):
        if len(self.memory) < batch_size: # if the memory is smaller than batch size then it wont optimize
            return 0.0, 0.0
        self.imagined_memory.clear() # memory for dynamic training

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size) # randomly samples a batch of batch_size from the memory

        state_action_batch = torch.cat((state_batch, action_batch.float()), dim=1) # concatenates state and action batches
        dynamic_output = self.dynamic(state_action_batch) # get the predicted reward, next state, and done from dynamic network
        dynamic_reward = dynamic_output[:, 0] # extract predicted reward
        dynamic_next_state = dynamic_output[:, 1:1+input_dimensions] # extract predicted next state
        dynamic_done = dynamic_output[:, -1] # extract predicted done
        
        dynamic_loss = nn.MSELoss()(dynamic_reward, reward_batch) + nn.MSELoss()(dynamic_next_state, next_state_batch) + nn.BCEWithLogitsLoss()(dynamic_done, done_batch) # calculates loss for dynamic network
        last_dynamic_loss = dynamic_loss.item()

        self.dynamic_optimizer.zero_grad()
        dynamic_loss.backward()
        self.dynamic_optimizer.step()
        
        # collect rollouts from current state batch
        self.imagine(state_batch)
        
        # use both imagined and real data for training critic
        half = batch_size // 2
        r_state, r_action, r_reward, r_next_state, r_done = self.memory.sample(half)
        i_state, i_action, i_reward, i_next_state, i_done = self.imagined_memory.sample(half)
        d_state_batch = torch.cat([r_state, i_state])
        d_action_batch = torch.cat([r_action, i_action])
        d_reward_batch = torch.cat([r_reward, i_reward])
        d_next_state_batch = torch.cat([r_next_state, i_next_state])
        d_done_batch = torch.cat([r_done, i_done])
        
        # train critic on new data TODO n-step truncation
        q_values = self.critic(d_state_batch).gather(1, d_action_batch).squeeze() # predicts q-values for all actions and extracts value of action actually taken

        with torch.no_grad(): # does not remember operations
            max_next_q_values = self.target(d_next_state_batch).max(1)[0] # outputs best possible q-value from next state
            target_q_values = d_reward_batch + gamma * max_next_q_values * (1-d_done_batch) # calculates immediate reward and future estimated reward
        
        critic_loss = nn.MSELoss()(q_values, target_q_values) # calculates distance from predicated q-values to target q-value
        last_critic_loss = critic_loss.item()

        self.critic_optimizer.zero_grad() # clears old gradients
        critic_loss.backward() # computes gradients of loss w.r.t. model parameters
        self.critic_optimizer.step() # updatse network weights 

        return last_critic_loss, last_dynamic_loss

    # dynamic model 
    def dynamic_step(self, state, action):
        state = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor([[action]]) # change to 1x1 tensor to match batch dimensions
        state_action = torch.cat((state.unsqueeze(0), action_tensor), dim=1) # concatenates state and action

        with torch.no_grad(): # disables tracking
            dynamic_output = self.dynamic(state_action) # uses dynamic model to predict next state, reward, and done

        next_state = dynamic_output[:, 1:1+input_dimensions].squeeze(0) # extracts predicted next state from output
        reward = dynamic_output[:, 0].item() # extracts predicted reward from first output dimension
        done = torch.sigmoid(dynamic_output[:, -1]).item() > 0.5 # extracts done from last output and uses sigmoid to squash to 0-1 and 0.5 threshold to split into bool

        return next_state, reward, done

    def plan(self, state, depth):
        if depth == 0: # base case with no lookahead steps left
            with torch.no_grad(): # disables tracking
                return self.critic(state.unsqueeze(0)).max().item() # estimates future value at current leaf state with critic instead of simulation
            
        best = float('-inf') # track best value across all actions
        for action in range(output_dimensions): # tests every action 
            next_state, reward, done = self.dynamic_step(state, action) # simulates taking action using dynamic model
            if done: 
                value = reward # if the episode ends the value is the reward
            else:
                value = reward + gamma * self.plan(next_state, depth-1) # otherwise recursively plans next state w gamma discount
            best = max(best, value)
        return best 

    def select_planned_action(self, state, depth=3):
        state_tensor = torch.FloatTensor(state) # converts state to tensor
        best_action = 0
        best_value = float('-inf')
        
        for action in range(output_dimensions): # tests every action possible from the root 
            next_state, reward, done = self.dynamic_step(state_tensor, action) 
            if done:
                value = reward
            else:
                value = reward + gamma * self.plan(next_state, depth-1)
            if value > best_value:
                best_value = value
                best_action = action # updates best value and best action
        return best_action
    
    def imagine(self, state_batch, epsilon=0.05):
        active_states = state_batch.clone().float()
        alive = torch.ones(len(state_batch), dtype=torch.bool) # tracks which rollouts are still running

        for _ in range(imagination_rollout_length):
            if not alive.any():
                break
            states = active_states[alive] # only active rollouts

            with torch.no_grad():
                actions = self.critic(states).argmax(dim=1) # one forward pass for all active states
                random_actions = torch.randint(0, output_dimensions, (len(states),)) # generates random tensor of floats
                mask = torch.rand(len(states)) < epsilon # compares to epsilon
                actions = torch.where(mask, random_actions, actions) # when mask is true take from random_actions, otherwise take from actions

                state_action = torch.cat([states, actions.float().unsqueeze(1)], dim=1)
                out = self.dynamic(state_action) # one forward pass for all

            rewards = out[:, 0]
            next_states = out[:, 1:1+input_dimensions]
            dones = torch.sigmoid(out[:, -1]) > 0.5

            for j in range(len(states)): # add all results into imagined memory
                self.imagined_memory.push(states[j].numpy(), actions[j].item(), rewards[j].item(), next_states[j].detach().numpy(), float(dones[j].item()))

            # update active_states and alive mask together
            alive_indices = alive.nonzero(as_tuple=False).squeeze(1)
            for j, orig_idx in enumerate(alive_indices):
                if dones[j]:
                    alive[orig_idx] = False
                else:
                    active_states[orig_idx] = next_states[j].detach()