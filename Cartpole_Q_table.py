
# coding: utf-8

# In[2]:





# In[22]:


import numpy as np
import gym
#Load the gym environment
env = gym.make('FrozenLake-v0')
#create an empty array with shape of the environment
Q = np.zeros([env.observation_space.n,env.action_space.n])
#initialize the parameters. Why are we setting d and y such as they are??
d = .7
y = .95
num_episodes = 2000
rList = []
# why are we using the for loop? 
for i in range(num_episodes):
    #set the initial observation
    s = env.reset()
    rAll = 0
    b = False
    j = 0
    while j < 99:
        j += 1
        # find the initial value of the action in the Q-table?
        a = np.argmax(Q[s,:]+np.random.randn(1,env.action_space.n)*(1./(i+1)))
        
        o,r,b,_=env.step(a)
        # this is the updating function for the Q-table                                  
        Q[s,a] = Q[s,a] + d*(r+(y*np.max(Q[o,:]-Q[s,a])))
        rAll += r
        s = o
        if b == True:
            break
    # I am curious about how this while loop and the for loop are working together, the reason behind implementing them.         
    rList.append(rAll)
    
print("cumulative reward values:")
print(rList)
print("Q table")
print(Q)                                          










