import nengo
import numpy as np
import gym

 
 
env = gym.make('MountainCar-v0')
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
        
        action = 0
        max = action0
        
        if action1 > max:
            action = 1
            max = action1
        if action2 > max:
            action = 2
            max = action2
        
        observation, reward, done, info = env.step(action)
        env.render()
        totalreward += reward
        if done:
            break
    return totalreward

##------------ Random Search Strategy ------------
bestparams = None  
bestreward = 0
steps = 0
acts = 3
obs = 2

for _ in range(10000):
    parameters = np.random.random((acts, obs))
    
    reward = run_episode(env,parameters)
    steps += 1
    
    if reward < bestreward:
        print("New Reward: ", reward)
        print("New params at step: ",steps)
        print("params0: ", parameters[0])
        print("params1: ", parameters[1])
        print("params2: ", parameters[2])
        
        
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward < 100:
            print()
            break  
        