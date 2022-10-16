# Pair programming: Eric Rosen, Jee Won Kyra Park
# Vanilla Policy Gradient on Cartpole.
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def reward_togo(episodes):
    # initialize our list of lists (transitions per episode)
    episodes_reward_togo = []
    for episode in episodes:
        episode_reward_togo = []
        returns = 0
        for transition in reversed(episode):
            returns += transition[2]
            new_transition = transition
            # update reward to reward-to-go.
            new_transition[2] = returns
            # for the current episode, add the updated transition.
            episode_reward_togo.append(new_transition)
        episodes_reward_togo.append(episode_reward_togo)
    return episodes_reward_togo

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.sm = nn.Softmax(dim=1)
      # First fully connected layer that takes in state (4D)
        self.fc1 = nn.Linear(4, 16)
      # Second fully connected layer that outputs our distribution over 2 actions
        self.fc2 = nn.Linear(16, 2)
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        # softmax for the distribution
        x = self.sm(x)
        return x

my_policy = Policy()
#test for one random
#action_probability = my_policy.forward(torch.tensor([[0, 1, 0.1, 1]]))
#print(action_probability)
#make the cartpole gym environment
cp_task = 'CartPole-v1'
ll_task = 'LunarLander-v2'
env = gym.make(cp_task, render_mode="human")


#TODO: NN that learns a value function (regression)   input: state, output: single number
# action - softmax learns a normal distribution (normalizes)
# value function - NN learns.
# Pytorch DQN

#number of policy updates
K = 1
#number of time steps
T = 100

for k in range(K):
    #list of transitions sampled for the policy
    #contains [observation, action, reward, observation']
    episodes = []
 
    #reset environment
    observation, info = env.reset()
    returns = 0
    episode = []
    for t in range(T):
        # compute action probability with the neural net.
        action_probability = my_policy.forward(torch.tensor([observation]))
        #take a random action. make the tensor into np array.
        action_probability = action_probability.cpu().detach().numpy()[0]
        print(action_probability)
        # sample an action according to the current probability distribution.
        action = random.choices(population=[0,1], weights=action_probability, k=1)[0]
        next_observation, reward, terminated, truncated, info = env.step(action)
        returns += reward
        #build transition
        transition = [observation, action, reward, next_observation, terminated]
        #add transition to list of transitions
        episode.append(transition)

        #if we completed an episode, reset the environment and observation
        if terminated:
          #  print("episode has terminated with retursn {R}".format(R=returns))
            observation, info = env.reset()
            returns = 0
            episodes.append(episode)
            episode = []
        else:
            #set current observation to new observation
            observation = next_observation
    #print(episodes[0])
    episodes_reward_togo = reward_togo(episodes)
    #print(episodes_reward_togo[0])

