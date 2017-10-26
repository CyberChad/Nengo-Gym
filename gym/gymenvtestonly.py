#import nengo
from time import sleep
import numpy as np
import gym

env = gym.make('CartPole-v0')

#env.configure(remote=1)
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    sleep(0.1)
env.render(close=True)
#
#import nengo.Simulator as model

#with model:
    
    