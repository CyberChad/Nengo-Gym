# Adaptive motor Control

# Now we use learning to improve our control of a system.  Here we have a
# randomly generated motor system that is exposed to some external forces
# (such as gravity acting on an arm).  The system needs to learn to adjust
# the commands it is sending to the arm to account for gravity.

# The basic system is a standard PD controller.  On top of this, we add
# a population where we use the normal output of the PD controller as the 
# error term.  This turns out to cause the system to learn to adapt to
# gravity (this is Jean-Jacques Slotine's dynamics learning rule).

# When you initially run this model, learning is turned off so you can see
# performance without learning.  Move the stop_learn slider down to 0 to
# allow learning to happen.

import gym
from gym import wrappers

import nengo
# requires ctn_benchmark https://github.com/ctn-waterloo/ctn_benchmarks
import ctn_benchmark.control as ctrl
import numpy as np

global env
global state
global steps
steps = 0
global total_reward
total_reward = 0

class Param:
    pass
p = Param()
p.D = 1
p.dt=0.001
p.seed=1
p.noise=0.1
p.Kp=2
p.Kd=1
p.Ki=0
p.tau_d=0.001
p.period=4
p.n_neurons=500
p.learning_rate=1
p.max_freq=1.0
p.synapse=0.01
p.scale_add=2
p.delay=0.00
p.filter=0.00
p.radius=1

# class CartPole(object):
#       
#     def __init__(self):
#         #super(NengoGymLunarLander, self).__init__(self.step)
#         #self.name = name
#         print("Gym CartPole Init")        
#         
#         self.feedback = []
#         self.controls = []
#         # self.size_in = size_in
#         # self.size_out = size_out
#         
#         self.env = gym.make("LunarLander-v2")
#         
#         print("Action Space:")
#         print(self.env.action_space)
#      
#         
#         print("Observation Space:")
#         print(self.env.observation_space)
#         print(self.env.observation_space.high)
#         print(self.env.observation_space.low)
#         
#         
#         self.reward = 0
#         self.total_reward = 0
#         self.steps = 0
#         # self.output = []
#         self.state = self.env.reset()
# 
# 
#     #handles the environment state and possible reward value passed back
#     #reinforce heuristics based on reward     
#     def handle_input(values):
#         return 0 #nothing for now
# 
# 
#     def handle_output(self):
#         return 0 #nothing for now
#         
#     def heuristic(self,state):
#         action = 0
#         #PID controller for trajectory optimizatio
#         #Engine control based on Pontryagin's maximum principle (full thrust on/off)
#         
#         angle_targ = state[0]*0.5 + state[2]*1.0         # angle should point towards center (state[0] is horizontal coordinate, state[2] hor speed)
#         if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
#         if angle_targ < -0.4: angle_targ = -0.4
#         hover_targ = 0.55*np.abs(state[0])           # target y should be proporional to horizontal offset
#     
#         # PID controller: state[4] angle, state[5] angularSpeed
#         angle_todo = (angle_targ - state[4])*0.5 - (state[5])*1.0
#         #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))
#     
#         # PID controller: state[1] vertical coordinate state[3] vertical speed
#         hover_todo = (hover_targ - state[1])*0.5 - (state[3])*0.5
#         #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))
#     
#         if state[6] or state[7]: # legs have contact
#             angle_todo = 0
#             hover_todo = -(state[3])*0.5  # override to reduce fall speed, that's all we need after contact
#         # 
#         # if self.env.continuous:
#         #     action = np.array( [hover_todo*20 - 1, -angle_todo*20] )
#         #     action = np.clip(action, -1, +1)
#         # else:
#         #     action = 0
#         
#         if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: action = 2
#         elif angle_todo < -0.05: action = 3
#         elif angle_todo > +0.05: action = 1
# 
#         return action
#         
#     def __call__(self, t, action):
#         #pre-processing sensor/feedback data        
#         #self.handle_input( values )
#         # print("Node _Call_")  
#         
#         #send next action event to the environment, receive feedback:
#         #state [vector] : of agent object in the environment (position, condition, etc)
#         #reward [scalar] : scalar feedback on meeting defined goal/error conditions
#         #done [bool] : if environment is finished running
#         #info [str] : debug info
#         
#         #action = self.heuristic(self.state) #static thrust for now
#         
#         #controls[0] = action[0] > 0
#         #controls[0] = action[0] >
#         input = 0
#         
#         #input = np.int(np.max(action))
#         
#         
#         
#         if action[1] > 0.5:
#             input = 3
#         elif action[1] < -0.5:
#             input = 1
#         elif action[0] > 0.5:
#             input = 2
#         else:
#             input = 0
#                 
#                
#         #input = np.int(action)
#         
#         # input 0 = no thrust (fall)
#         # input 1 = right thrust (go left)
#         # input 2 = bottom thrust (go up)
#         # input 3 = left thrust (go right)
#         
#         self.state, self.reward, done, info = self.env.step(input) #
#         #env.step(action) 
# 
#         # if values[0] > 0: action=1 #left thruster
#         # elif x[0] < 0: action=3 #right thruster
#         # else: action=0 #do nothing
# 
# 
#         self.env.render() #one frame
#         
#         #tally reward for epoch updates
#         self.total_reward += self.reward
#         #total_reward += 1
#         
#         newstate = []    
#         newstate = self.state
#         
#         #newstate[4] = newstate[4]/6
#         #newstate[5] = newstate[5]/6        
#         
#         #preliminary reward function logging
#         # if self.steps % 20 == 0 or done:
#         #     print(["{:+0.2f}".format(x) for x in self.state])
#         #     print("step {} total_reward {:+0.2f}".format(self.steps, self.total_reward))
#         # 
#          #preliminary reward function logging
#         if self.steps % 20 == 0 or done:
#             print(["{:+0.2f}".format(x) for x in newstate])
#             print("step {} total_reward {:+0.2f}".format(self.steps, self.total_reward))       
#         #increment counter for learning rate
#         self.steps += 1
#         
#         #check to see if we have crashed, landed, etc
#         if done:
#             #env.render(close=True)
#             #raise Exception("Simulation done")
#             self.state = self.env.reset()
#         
# 
#         
#         return newstate

def gymrep(t, x):
    
    global env
    global state
    global steps
    global total_reward
    
    if state[2] > 0:
        action = 1
    else:
        action = 0
    #input = np.int(x[)
    
    #action = 1
    
    state, reward, done, info = env.step(action)
    total_reward += reward
    env.render() #one frame
          
    print(["{:+0.2f}".format(x) for x in state])
    
    if steps % 20 == 0 or done:
        print(["{:+0.2f}".format(x) for x in state])
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))       
    #increment counter for learning rate
    steps += 1
    
    if done:
        #env.render(close=True)
        #raise Exception("Simulation done")
        state = env.reset()
    return 0
        
model = nengo.Network()
with model:
    
    global env
    
    

    env = gym.make('CartPole-v0')
    state = env.reset()
    gymNode = nengo.Node(gymrep, size_in=1, size_out=1)
    
    print("Action Space:")
    print(env.action_space)
    
    
    
    print("Observation Space:")
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)     

parameters = np.random.rand(4) * 2 - 1


    system = ctrl.System(p.D, p.D, dt=p.dt, seed=p.seed,
            motor_noise=p.noise, sense_noise=p.noise,
            scale_add=p.scale_add,
            motor_scale=10,
            motor_delay=p.delay, sensor_delay=p.delay,
            motor_filter=p.filter, sensor_filter=p.filter)

    def minsim_system(t, x):
        return system.step(x)

    minsim = nengo.Node(minsim_system, size_in=p.D, size_out=p.D,
                        label='minsim')

    state_node = nengo.Node(lambda t: system.state, label='state')

    pid = ctrl.PID(p.Kp, p.Kd, p.Ki, tau_d=p.tau_d)
    
    
    #define the controls going in. In this case, D*2 = left/right
    control = nengo.Node(lambda t, x: pid.step(x[:p.D], x[p.D:]),
                         size_in=p.D*2, label='control')
    
    nengo.Connection(minsim, control[:p.D], synapse=0)

    adapt = nengo.Ensemble(p.n_neurons, dimensions=p.D,
                           radius=p.radius, label='adapt')
    
    nengo.Connection(minsim, adapt, synapse=None)
    
    motor = nengo.Ensemble(p.n_neurons, p.D, radius=p.radius)
    nengo.Connection(motor, minsim, synapse=None)
    
    nengo.Connection(motor, gymNode)
    
    nengo.Connection(control, motor, synapse=None)
    
    conn = nengo.Connection(adapt, motor, synapse=p.synapse,
            function=lambda x: [0]*p.D,
            learning_rule_type=nengo.PES(1e-4 * p.learning_rate))

    error = nengo.Ensemble(p.n_neurons, p.D)
    nengo.Connection(control, error, synapse=None,
                        transform=-1)
    nengo.Connection(error, conn.learning_rule)
    

    signal = ctrl.Signal(p.D, p.period, dt=p.dt, max_freq=p.max_freq, seed=p.seed)
    
    
    #this is the desired position. We want to stay at 0 (pole straight up)
    desired = nengo.Node(signal.value, label='desired')
    
    nengo.Connection(desired, control[p.D:], synapse=None)
    
    
    stop_learn = nengo.Node([1])
    nengo.Connection(stop_learn, error.neurons, transform=np.ones((p.n_neurons,1))*-10)
    
    
    result = nengo.Node(None, size_in=p.D*2)
    
    nengo.Connection(desired, result[:p.D], synapse=None)
    nengo.Connection(minsim, result[p.D:], synapse=None)
    
sim = nengo.Simulator(model)
sim.run(5)