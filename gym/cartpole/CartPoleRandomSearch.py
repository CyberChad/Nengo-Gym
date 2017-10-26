import nengo
import numpy as np
import gym

 
 
env = gym.make('CartPole-v0')
env.reset()

print("Environment Action Space: ",env.action_space)
print("Enviornment Observation Space: ",env.observation_space)

def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
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
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    steps += 1
    
    if reward > bestreward:
        print("New params at step ",steps,": ", parameters)
        print("New Reward: ", reward)
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break  
        