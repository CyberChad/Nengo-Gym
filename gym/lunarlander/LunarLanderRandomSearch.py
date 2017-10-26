import nengo
import numpy as np
import gym

 
 
env = gym.make('LunarLander-v2')
env.reset()

print("Environment Action Space: ",env.action_space)
print("Enviornment Observation Space: ",env.observation_space)
print("Enviornment Observation High: ",env.observation_space.high)
print("Enviornment Observation Low: ",env.observation_space.low)

def normalize(high, low, value):
    #normalize observations; replace with automatic transform
    range = np.abs(high) + np.abs(low)
    adj = -(high - range/2)
    factor = 2/range
    
    value += adj
    value *= factor
    
    return value

def run_episode(env, parameters):  
    observation = env.reset()
    
    
    #observation[0] = normalize(0.6,-1.2,observation[0])
    #observation[1] = normalize(0.07,-0.07,observation[1])
    
    totalreward = 0
    for _ in range(200):
        action0 = np.matmul(parameters[0],observation)
        action1 = np.matmul(parameters[1],observation)
        action2 = np.matmul(parameters[2],observation)
        action3 = np.matmul(parameters[3],observation)
        
        action = 0
            
        if action1 > action0:
            action = 1
        if action2 > action1:
            action = 2
        if action3 > action2:
            action = 3
        
        observation, reward, done, info = env.step(action)
        env.render()
        totalreward += reward
        if done:
            break
    return totalreward

##------------ Random Search Strategy ------------
bestparams = None  
bestreward = -999999
steps = 0
acts = 4
obs = 8

for _ in range(10000):
    parameters = np.random.random((acts, obs))
    
    reward = run_episode(env,parameters)
    
    #print("round: ",steps," reward: ",reward)
    
    steps += 1
    
    if reward > bestreward:
        print("Steps: ",steps,"Reward: ", reward)
        print("New params at step: ",steps)
        print("params0: ", parameters[0])
        print("params1: ", parameters[1])
        print("params2: ", parameters[2])
        print("params3: ", parameters[3])
        
        
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward > 200:
            print()
            break  
        