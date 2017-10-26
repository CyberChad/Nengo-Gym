#
# The following code liberally taken from the OpenAI gym/examples/random_agent.py source
#

import argparse
import logging
import sys
import random

import numpy as np
import nengo

import gym
from gym import wrappers

#************** Agent Classes *****************


class NengoAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

#*********** BEGIN MAIN *************

# if __name__ == '__main__':
#     
#     #Accept environments to test. See 
#     parser = argparse.ArgumentParser(description=None)
#     parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
#     args = parser.parse_args()
# 
#     # Call `undo_logger_setup` if you want to undo Gym's logger setup
#     # and configure things manually. (The default should be fine most
#     # of the time.)
#     gym.undo_logger_setup()
#     logger = logging.getLogger()
#     formatter = logging.Formatter('[%(asctime)s] %(message)s')
#     handler = logging.StreamHandler(sys.stderr)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
# 
#     # You can set the level to logging.DEBUG or logging.WARN if you
#     # want to change the amount of output.
#     logger.setLevel(logging.INFO)
# 
# 
#     #Create the Environment. See agentapi.py for Nengo's get/set defs.
#     env = gym.make(args.env_id)
# 
#     # You provide the directory to write to (can be an existing
#     # directory, including one with existing data -- all monitor files
#     # will be namespaced). You can also dump to a tempdir if you'd
#     # like: tempfile.mkdtemp().
#     outdir = '/tmp/nengo-gym-agent-results'
#     env = wrappers.Monitor(env, directory=outdir, force=True)
#     env.seed(0)
#     
#     #---------------------- Environment Detection --------------------
#     
#     #the action_space is the list of controls we can interface with.
#     #we can sample from the Space or check that something belongs to it.
#     #This is good for random environments.
#     
#     from gym import spaces
#     space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
#     x = space.sample()
#     assert space.contains(x)
#     assert space.n == 8
#     
#     print("Action space: {}".format(space.n))
#     
#     agent = NengoAgent(env.action_space)
#     
#     #the action
# 
#     episode_count = 100
#     reward = 0
#     done = False
# 
#     for i in range(episode_count):
#         ob = env.reset()
#         while True:
#             action = agent.act(ob, reward, done)
#             ob, reward, done, _ = env.step(action)
#             if done:
#                 break
#             # Note there's no env.render() here. But the environment still can open window and
#             # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
#             # Video is not recorded every episode, see capped_cubic_video_schedule for details.
# 
#     # Close the env and write monitor result info to disk
#     env.close()
# 
#     # Upload to the scoreboard. We could also do this from another
#     # process if we wanted.
#     
#     gym.scoreboard.api_key="sk_Ly0KReUFRsWx4PXiBMllnA"
#     
#     gym.upload(outdir)

#Should embed the Nengo Agent into the Nengo Model {agent->model->sim}
#This structure borrowed from robots.py example
#class NengoGym(object):
    #def __init__(self, sim_dt=0.05, nengo_dt=0.001, sync=True):

# class PID(object):
#     def __init__(self, dimensions=1):
#         self.last_error = np.zeros(dimensions)
#         return 0
#         _
#     def step(self):
#         return 0
# 
# class PIDNode(nengo.Node):
#     def __init__(self, dimensions, **kwargs):
#         self.dimensions = dimensions
#         self.pid = PID(dimensions=dimensions, **kwargs)
#         super(PIDNode, self).__init__(self.step, size_in=dimensions*4)
#         
#     def step(self, t, values):
#         #do some correction stuff
#         #return self.pid.step(some control correction values)
#         return 0
# 

class NengoGymLunarLander(object):
    
    def __init__(self):
        #super(NengoGymLunarLander, self).__init__(self.step)
        #self.name = name
        print("Gym Init")        
        
        
        
        self.feedback = []
        self.controls = []
        # self.size_in = size_in
        # self.size_out = size_out
        
        self.env = gym.make("LunarLander-v2")
        
        print("Action Space:")
        print(self.env.action_space)
     
        
        print("Observation Space:")
        print(self.env.observation_space)
        print(self.env.observation_space.high)
        print(self.env.observation_space.low)
        
        #write some transform to dynamically generate parameter matrix        
        
        self.reward = 0
        self.total_reward = 0
        self.steps = 0
        # self.output = []
        self.state = self.env.reset()


    #handles the environment state and possible reward value passed back
    #reinforce heuristics based on reward     
    def handle_input(values):
        return 0 #nothing for now


    def handle_output(self):
        return 0 #nothing for now
        
 
        
    def __call__(self, t, control):

        action = 0
        
        if control[1] > control[0]:
            action = 1
        if control[2] > control[1]:
            action = 2
        if control[3] > control[2]:
            action = 3

        self.state, self.reward, done, info = self.env.step(action) #
        #wait(200)
        #env.step(action) 

        self.env.render() #one frame
        
        #tally reward for epoch updates
        self.total_reward += self.reward
        #total_reward += 1
    
        if self.steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in self.state])
            print("step {} total_reward {:+0.2f}".format(self.steps, self.total_reward))       
        #increment counter for learning rate
        self.steps += 1
        
        #check to see if we have crashed, landed, etc
        if done:
            #env.render(close=True)
            #raise Exception("Simulation done")
            self.steps = 0
            self.state = self.env.reset()
            self.total_reward = 0
        
        self.state[2] /=2
        self.state[3] /=2
        self.state[4] /=2
        self.state[5] /=2

        self.state[4] /=2
        self.state[5] /=2
        
        return self.state
        
class NengoGymNode(nengo.Node):       

    def __init__(self, **kwargs):
        self.env = NengoGymLunarLander(**kwargs)
        
        def func(t, x):
            return self.env.step(x[0])
        
        super(NengoGymNode, self).__init__(size_in=1, size_out=1)

def heuristic(state):
     actions = np.zeros(4)
     actions[0] = 0.5
     #PID controller for trajectory optimizatio
     #Engine control based on Pontryagin's maximum principle (full thrust on/off)
     
     angle_targ = state[0]*0.5 + state[2]*1.0         # angle should point towards center (state[0] is horizontal coordinate, state[2] hor speed)
     if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
     if angle_targ < -0.4: angle_targ = -0.4
     hover_targ = 0.55*np.abs(state[0])           # target y should be proporional to horizontal offset
 
     # PID controller: state[4] angle, state[5] angularSpeed
     angle_todo = (angle_targ - state[4])*0.5 - (state[5])*1.0
     print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))
 
     # PID controller: state[1] vertical coordinate state[3] vertical speed
     hover_todo = (hover_targ - state[1])*0.5 - (state[3])*0.5
     print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))
 
     if (state[6]>0) or (state[7]>0): # legs have contact
         angle_todo = 0
         hover_todo = -(state[3])*0.5  # override to reduce fall speed, that's all we need after contact
     # 
     # if self.env.continuous:
     #     action = np.array( [hover_todo*20 - 1, -angle_todo*20] )
     #     action = np.clip(action, -1, +1)
     # else:
     #     action = 0
     
    #  if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: actions[2] = 1
    #  elif angle_todo < -0.05: actions[3] = 1
    #  elif angle_todo > +0.05: actions[1] = 1 
     
     actions[2] = hover_todo
     actions[3] = -angle_todo
     actions[1] = angle_todo 
 
     return actions  
    

model = nengo.Network()

with model:
   
 
   #pid = PIDNode(dimensions=1)
    environment = NengoGymLunarLander()

    env = nengo.Node(environment, size_in=4, size_out=8)
    #correction = nengo.Node(None, size_in=1, size_out=4)
   
    #sensors = nengo.Ensemble(n_neurons=500, dimensions=8, neuron_type=nengo.Direct())                       
    #sensors = nengo.Ensemble(n_neurons=500, dimensions=2)
    #controls = nengo.Ensemble(n_neurons=500, dimensions=4, neuron_type=nengo.Direct())

    sensors = nengo.Ensemble(n_neurons=500, dimensions=8, neuron_type=nengo.Direct())                       
    #sensors = nengo.Ensemble(n_neurons=500, dimensions=2)
    controls = nengo.Ensemble(n_neurons=500, dimensions=4, neuron_type=nengo.Direct())
    
    nengo.Connection(env, sensors)
    nengo.Connection(sensors, controls, function=heuristic, synapse=None)
    nengo.Connection(controls, env, synapse=0)
    #nengo.Connection(correction, env)
    
    manual = nengo.Node(0)
    #nengo.Connection(manual,env)
    

sim = nengo.Simulator(model)
sim.run(5)

