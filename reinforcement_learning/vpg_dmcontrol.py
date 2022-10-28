# Code based on dm_control's tutorial.ipynb
# Pair programming: Eric Rosen, Jee Won Kyra Park
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

#@title Run to install MuJoCo and `dm_control`
import distutils.util
import subprocess
if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')
# Configure dm_control to use the EGL rendering backend (requires GPU)
#env MUJOCO_GL=egl
#make sure to disable the headless option for matplotlib backend to run locally.
print('Checking that the dm_control installation succeeded...')
try:
  from dm_control import suite
  env = suite.load('cartpole', 'swingup')
  pixels = env.physics.render()
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".')
else:
  del pixels, suite

# The basic mujoco wrapper.
from dm_control import mujoco
# import the rl module
from dm_control.rl import control
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Go to target and run through corridor examples
from dm_control.locomotion.tasks import go_to_target
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks
from dm_control.locomotion.walkers import base
# General
import copy
import os
import itertools
import numpy as np

#testing
import mock

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import PIL.Image
# To define goal
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions

# Inline video helper function
if os.environ.get('COLAB_NOTEBOOK_TEST', False):
  # We skip video generation during tests, as it is quite expensive.
  display_video = lambda *args, **kwargs: None
else:
  def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    # commented out to not use in colab.
    #matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    #return HTML(anim.to_html5_video())
    plt.show()
    return

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
        self.sm = nn.Softmax(dim=0)
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
        return 

policy = Policy()
#action_probability = policy.forward(torch.tensor([1,2,3,4], dtype=torch.float))
# Seed numpy's global RNG so that cell outputs are deterministic. We also try to
# use RandomState instances that are local to a single cell wherever possible.
np.random.seed(42)


# use the benchmark control suite
max_len = max(len(d) for d, _ in suite.BENCHMARKING)
for domain, task in suite.BENCHMARKING:
  print(f'{domain:<{max_len}}  {task}')

#@title Loading and simulating a `suite` task{vertical-output: true}

# Load the environment
random_state = np.random.RandomState(42)
env = suite.load('hopper', 'stand', task_kwargs={'random': random_state})

# Simulate episode with random actions
duration = 6  # Seconds
frames = []
ticks = []
rewards = []
observations = []

spec = env.action_spec()
time_step = env.reset()

#TODO: NN that learns a value function (regression)   input: state, output: single number
# action - softmax learns a normal distribution (normalizes)
# value function - NN learns.
# Pytorch DQN
K=1
T = 100
for k in range(K):
    #list of transitions sampled for the policy
    #contains [observation, action, reward, observation']
    episodes = []
 
    #reset environment
    #observation, info = env.reset()
    time_step = env.reset()
    
    returns = 0
    episode = []
    for t in range(T):
        # compute action probability with the neural net.
       
        observation = np.array(time_step.observation.items())
        print(observation)
        action_probability = policy.forward(torch.tensor([observation]))
        #take a random action. make the tensor into np array.
        action_probability = action_probability.cpu().detach().numpy()[0]
        print(action_probability)
        # sample an action according to the current probability distribution.
        action = random.choices(population=[0,1], weights=action_probability, k=1)[0]
        #next_observation, reward, terminated, truncated, info = env.step(action)
        env.step(action)
        returns += time_step.reward
        #build transition
        transition = [observation, time_step.action, time_step.reward, time_step.observation, time_step.terminated]
        #add transition to list of transitions
        episode.append(transition)

        #if we completed an episode, reset the environment and observation
        if time_step.terminated:
          #  print("episode has terminated with retursn {R}".format(R=returns))
            time_step = env.reset()
            returns = 0
            episodes.append(episode)
            episode = []
        else:
            #set current observation to new observation
            observation = time_step.observation
    #print(episodes[0])
    episodes_reward_togo = reward_togo(episodes)
    

'''
html_video = display_video(frames, framerate=1./env.control_timestep())

# Show video and plot reward and observations
num_sensors = len(time_step.observation)

_, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
ax[0].plot(ticks, rewards)
ax[0].set_ylabel('reward')
ax[-1].set_xlabel('time')

for i, key in enumerate(time_step.observation):
  data = np.asarray([observations[j][key] for j in range(len(observations))])
  ax[i+1].plot(ticks, data, label=key)
  ax[i+1].set_ylabel(key)

html_video




# define a goal for this rl environment
#physics = mujoco.Physics.from_xml_string(static_model)
walker = cmu_humanoid.CMUHumanoid()
task = go_to_target.GoToTarget(walker, arena=floors.Floor())
model = control.Environment(walker, task)
pixels = walker.render()
PIL.Image.fromarray(pixels)
'''