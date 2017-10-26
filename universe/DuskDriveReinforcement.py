import gym
import universe  # register the universe environments

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation = env.reset()


def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
      
      action = [[('KeyEvent', 'ArrowUp', True)] for ob in observation]  # your agent here
      observation, reward_n, done_n, info = env.step(action)
      env.render()

      totalreward += reward
      if done:
          break
    return totalreward

##------------ Hill-Climbing Strategy -------------        
        
noise_scaling = 0.7  
parameters = np.random.rand(3) * 2 - 1  
bestreward = 0
episodes_per_update = 10
steps = 0
# 
# for _ in range(10000):  
#     newparams = parameters + (np.random.rand(3) * 2 - 1)*noise_scaling
#     reward = 0  
#     for _ in range(episodes_per_update):  
#         run = run_episode(env,newparams)
#         reward += run
#     steps += 1
#     if reward > bestreward:
#         print("New params at step ",steps,": ", parameters)
#         print("New Reward: ", reward)
#         bestreward = reward
#         parameters = newparams
#         if reward >= 200:
#             break  

while True:
  # agent which presses the Up arrow 60 times per second
  action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()
