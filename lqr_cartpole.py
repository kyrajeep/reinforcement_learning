# Pair programming with Eric Rosen
import gym
import control
import numpy as np
def lqr_policy(observation):
    # observation: state. 4D in this case.
    x, v, theta, v_theta = observation
    # cost function 
    Q = R = np.identity(4)
    # linearization
    A = np.identity(4)
    B = np.ones((4,1))
    #K (2-d array) – State feedback gains: ???
    #S (2-d array) – Solution to Riccati equation
    #E (1-d array) – Eigenvalues of the closed loop system
    K, S, E = control.lqr(A,B,Q,R)
    
    return    

env = gym.make('CartPole-v1')
observation = env.reset()
total_reward = 0
for _ in range(50):
    env.render()
    # env.step() takes an action and returns all we need to know
    observation, reward, done, info = env.step(lqr_policy(observation))
    #print(reward)
    total_reward += reward
print(total_reward)
