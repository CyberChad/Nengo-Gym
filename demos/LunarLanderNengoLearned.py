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

        action = np.argmax(control)

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
        

        
        return self.state
        
class NengoGymNode(nengo.Node):       

    def __init__(self, **kwargs):
        self.env = NengoGymLunarLander(**kwargs)
        
        def func(t, x):
            return self.env.step(x[0])
        
        super(NengoGymNode, self).__init__(size_in=1, size_out=1)

def heuristic(observation):
    
    actions = np.zeros(4)
    observation = observation.copy()
    #observation[1] *= 2
    #original Hillclimb with Simulated Annealing (linear noise reduction)
    #parameters = np.array([[-0.99784033, -1.48224947, -1.128918,   -1.81921602, -0.86971125, -0.95218424, -1.36412427, -0.39022652],
    
    #CP Corrected for touchtown sensitivity
    #parameters = np.array([[-0.99784033, -1.48224947, -1.128918,   -1.81921602, -0.86971125, -0.95218424, 2, 2],
    
    #CP Countering abnormal horiz/vert velocity upon entry
    parameters = np.array([\
    [-0.99784033, -1.48224947, 1.128918,   1.81921602, -0.86971125, -0.95218424, 2, 2], #do nothing
    [ 2.33565398,  2.71445419,  1.96071832,  2.03839483,  1.38366366,  1.29056097,  1.95440862,  1.69941021], #steer left
    [ 0.86654426,  0.99100604,  0.3657774,  -0.59456957, -0.27813512, -0.05787973,  0.97960713,  0.97284944], #go up
    [ 1.36627608,  1.81421544,  0.23252924,  0.3666804,   1.19115073,  0.94803147,  0.16546737,  0.18223217]]) #steer right
    
    actions[0] = np.matmul(parameters[0],observation)
    actions[1] = np.matmul(parameters[1],observation)
    actions[2] = np.matmul(parameters[2],observation) 
    actions[3] = np.matmul(parameters[3],observation)
    #print("actions: ",actions)
    
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

    sensors = nengo.Ensemble(n_neurons=2000, dimensions=8, neuron_type=nengo.LIFRate(), radius=2)                       
    #sensors = nengo.Ensemble(n_neurons=500, dimensions=2)
    controls = nengo.Ensemble(n_neurons=500, dimensions=4, neuron_type=nengo.Direct())
    
    nengo.Connection(env, sensors, synapse=None)
    nengo.Connection(sensors, controls, function=heuristic, synapse=0)
    nengo.Connection(controls, env, synapse=None)
    #nengo.Connection(correction, env)
    
    manual = nengo.Node(0)
    #nengo.Connection(manual,env)
    

sim = nengo.Simulator(model)
sim.run(5)

