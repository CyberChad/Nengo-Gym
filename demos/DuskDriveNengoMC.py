#
# The following code liberally taken from the OpenAI gym/examples/random_agent.py source
#

import argparse
import logging
import sys

import numpy as np
import nengo

import gym
import universe
from gym import wrappers

#from nengo.processes import WhiteSignal #replace this with input from Env.()




class NengoUniverse(object):
    
          
    def __init__(self):
        #super(NengoGymLunarLander, self).__init__(self.step)
        #self.name = name
        print("*** Universe Init****")        
 
        
        self.env = gym.make('flashgames.DuskDrive-v0') # any Universe environment ID here
        self.env.configure(remotes=1)  # automatically creates a local docker container
        self.observation = self.env.reset()
        self.action = [[('KeyEvent', 'ArrowUp', True)] for ob in self.observation]
        self.actionstate = 0
        # print("Action Space:")
        # print(self.env.action_space)
        # 
        # print("Observation Space:")
        # print(self.env.observation_space)
        # print(self.env.observation_space.high)
        # print(self.env.observation_space.low)
        
        
        self.reward = 0
        self.total_reward = 0
        self.steps = 0
        # self.output = []
    #handles the environment state and possible reward value passed back
    #reinforce heuristics based on reward     
    def handle_input(values):
        return 0 #nothing for now


    def handle_output(self):
        return 0 #nothing for now
        
    def heuristic(self,state):

        return 0 #nothing for now
        
    def __call__(self, t, action_temp):
        
        #action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
        
        # if action_temp[0] < -0.3 and self.actionstate != 0:
        #     self.actionstate = 0
        #     self.action = [[('KeyEvent', 'ArrightRight', False)]]
        #     self.env.step(self.action)
        #     self.action = [[('KeyEvent', 'ArrowUp', False)]]
        #     self.env.step(self.action)
        #     self.action = [[('KeyEvent', 'ArrowLeft', True)]  for ob in self.observation]        
        #     self.observation, reward, done, info = self.env.step(self.action)           
        # elif action_temp[0] > 0.3 and self.actionstate != 1:
        #     self.actionstate = 1
        #     self.action = [[('KeyEvent', 'ArrowLeft', False)]]
        #     self.env.step(self.action)
        #     self.action = [[('KeyEvent', 'ArrowUp', False)]]
        #     self.env.step(self.action)
        #     self.action = [[('KeyEvent', 'ArrowRight', True)] for ob in self.observation]        
        #     self.observation, reward, done, info = self.env.step(self.action)
        # else: #default to gas
        #     self.actionstate = 2
        #     self.action = [[('KeyEvent', 'ArrowRight', False)]]
        #     self.env.step(self.action)
        #     self.action = [[('KeyEvent', 'ArrowLeft', False)]]
        #     self.env.step(self.action)
        # 
        self.action = [[('KeyEvent', 'ArrowUp', True)] for ob in self.observation]
        self.observation, reward, done, info = self.env.step(self.action)
        
        self.env.render() #one frame
        
        #tally reward for epoch updates
        self.total_reward += self.reward
        #total_reward += 1
        
        self.steps += 1
        
        #check to see if we have crashed, landed, etc
        # if done:
        #     env.render(close=True)
        #     raise Exception("Simulation done")
        #     self.state = self.env.reset()
        # 

        
        return self.observation
        
class NengoGymNode(nengo.Node):       

    def __init__(self, **kwargs):
        self.env = NengoGymLunarLander(**kwargs)
        
        def func(t, x):
            return self.env.step(x[0])
        
        super(NengoGymNode, self).__init__(size_in=1, size_out=1)

def heuristic(state):

#     type(state)
    
#     #PID controller for trajectory optimizatio
#     #Engine control based on Pontryagin's maximum principle (full thrust on/off)
    
#     angle_targ = state[0]*0.5 + state[2]*1.0         # angle should point towards center (state[0] is horizontal coordinate, state[2] hor speed)
#     #if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
#     #if angle_targ < -0.4: angle_targ = -0.4

#     if angle_targ >  0: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
#     if angle_targ < -0: angle_targ = -0.4


#     hover_targ = 0.55*np.abs(state[0])           # target y should be proporional to horizontal offset
    
#     # PID controller: state[4] angle, state[5] angularSpeed
#     angle_todo = (angle_targ - state[4])*0.5 - (state[5])*1.0
#     #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))
    
#     # PID controller: state[1] vertical coordinate state[3] vertical speed
#     hover_todo = (hover_targ - state[1])*0.5 - (state[3])*0.5
#     #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))
    
#     if state[6] or state[7]: # legs have contact
#         angle_todo = 0
#         hover_todo = -(state[3])*0.5  # override to reduce fall speed, that's all we need after contact
    
#     # # analog feedback
#     # action = np.array( [hover_todo*20 - 1, -angle_todo*20] )
#     # #action = np.clip(action, -1, +1)
#     # 
#     # #action = np.array([hover_todo,angle_todo])
#     # action = np.array([angle_todo,hover_todo])
    
#     action = [hover_todo,angle_todo]
#     #digital feedback
#     # if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
#     #     action[1] = 1
#     #     #action[0] = -1
#     # if angle_todo < -0.05: 
#     #     action[0] = -1
#     #     #action[1] = -1
#     # if angle_todo > +0.05: 
#     #     action[0] = 1
     action = 0
     return action

model = nengo.Network()

with model:
   
 
   #pid = PIDNode(dimensions=1)
    environment = NengoUniverse()

    env = nengo.Node(environment, size_in=1, size_out=4)
    correction = nengo.Node(None, size_in=1, size_out=4)
   
    sensors = nengo.Ensemble(n_neurons=1, dimensions=4, neuron_type=nengo.Direct())                       
    controls = nengo.Ensemble(n_neurons=1, dimensions=1)
    
    nengo.Connection(env, sensors)
    nengo.Connection(sensors, controls, function=heuristic, synapse=0)
    nengo.Connection(controls, correction)
    #nengo.Connection(correction, env)
    
    manual = nengo.Node(0)
    #nengo.Connection(manual,env)
    
    



sim = nengo.Simulator(model)
sim.run(5)

