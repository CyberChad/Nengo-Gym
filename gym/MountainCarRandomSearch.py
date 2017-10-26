import nengo
import numpy as np
import gym

 
 
env = gym.make('MountainCar-v0')
env.reset()

def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        
        if np.matmul(parameters,observation) < 0
            action = 1
        elif else 2
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
for _ in range(10000):  
    parameters = np.random.rand(2) * 2 - 1
    reward = run_episode(env,parameters)
    steps += 1
    if reward > bestreward:
        print("Step: ",steps)
        print("parameters: ",parameters)
        print("Reward: ",reward)
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break