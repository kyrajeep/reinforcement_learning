import gym
env = gym.make('CartPole-v1')
observation = env.reset()
total_reward = 0
for _ in range(50):
    env.render()
     # pick a random action
    observation, reward, done, info = env.step(env.action_space.sample())
    print(observation)
    total_reward += reward
print(total_reward)