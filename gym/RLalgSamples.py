import nengo
import numpy as np
import gym

 
 
env = gym.make('CartPole-v0')
env.reset()

#Hillclimb Search Alogirthm
parameters = np.random.rand(4) * 2 - 1
action = 0 if np.matmul(parameters,observation) < 0 else 1  

def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

##------------ Random Search Strategy ------------
bestparams = None  
bestreward = 0  
for _ in xrange(10000):  
    parameters = np.random.rand(4) * 2 - 1
    reward = run_episode(env,parameters)
    if reward > bestreward:
        bestreward = reward
        bestparams = parameters
        # considered solved if the agent lasts 200 timesteps
        if reward == 200:
            break
        
##------------ Hill-Climbing Strategy -------------        
        
noise_scaling = 0.1  
parameters = np.random.rand(4) * 2 - 1  
bestreward = 0  
for _ in xrange(10000):  
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling
    reward = 0  
    for _ in xrange(episodes_per_update):  
        run = run_episode(env,newparams)
        reward += run
    if reward > bestreward:
        bestreward = reward
        parameters = newparams
        if reward == 200:
            break        

#******************** Policy/Value Gradeint for High-Dimensional Online Learning!!!!        
#-------------- Policy Gradient with Tensorflow -------------
def policy_gradient():  
    params = tf.get_variable("policy_parameters",[4,2])
    state = tf.placeholder("float",[None,4])
    actions = tf.placeholder("float",[None,2])
    linear = tf.matmul(state,params)
    probabilities = tf.nn.softmax(linear)
    good_probabilities = tf.reduce_sum(tf.mul(probabilities, actions),reduction_indices=[1])
    # maximize the log probability
    log_probabilities = tf.log(good_probabilities)
    loss = -tf.reduce_sum(log_probabilities)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

##-------------- Value Gradient with Tensorflow -----------------
        
def value_gradient():  
    # sess.run(calculated) to calculate value of state
    state = tf.placeholder("float",[None,4])
    w1 = tf.get_variable("w1",[4,10])
    b1 = tf.get_variable("b1",[10])
    h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
    w2 = tf.get_variable("w2",[10,1])
    b2 = tf.get_variable("b2",[1])
    calculated = tf.matmul(h1,w2) + b2

    # sess.run(optimizer) to update the value of a state
    newvals = tf.placeholder("float",[None,1])
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)


model = nengo.Network()    
with model: 
    #Ensembles to represent populations
    pre = nengo.Ensemble(50, dimensions=1)
    post = nengo.Ensemble(50, dimensions=1)
    error = nengo.Ensemble(100, dimensions=1)
    actual_error = nengo.Ensemble(100, dimensions=1, neuron_type=nengo.Direct())
    
    #Actual Error = pre - post (direct mode)
    #Square
    nengo.Connection(pre, actual_error, function=lambda x: x**2, transform=-1)   
    #nengo.Connection(pre, actual_error, transform=-1)   #Communication Channel
    nengo.Connection(post, actual_error, transform=1)
     
    #Error = pre - post
    #Square
    nengo.Connection(pre, error, function=lambda x: x**2, transform=-1)
    #Communication Channel
    #nengo.Connection(pre, error, transform=-1, synapse=0.02)
    nengo.Connection(post, error, transform=1, synapse=0.02)
    
    #Connecting pre population to post population (communication channel)
    conn = nengo.Connection(pre, post, function=lambda x: np.random.random(1),
                            solver=nengo.solvers.LstsqL2(weights=True))
    
    #Adding the learning rule to the connection 
    conn.learning_rule_type ={'my_pes': nengo.PES(learning_rate=1e-3), 
                                        'my_bcm': nengo.BCM()}

    #Error connections don't impart current
    error_conn = nengo.Connection(error, conn.learning_rule['my_pes'])
    
    #Providing input to the model
    #input = nengo.Node(WhiteSignal(30, high=10))     # RMS = 0.5 by default
    input = nengo.Node(None, size_out = 1)     # RMS = 0.5 by default
    # Connecting input to the pre ensemble
    nengo.Connection(input, pre, synapse=0.02)  
    
    #function to inhibit the error population after 25 seconds
    def inhib(t):
        return 2.0 if t > 15.0 else 0.0
    
    #Connecting inhibit population to error population
    inhibit = nengo.Node(inhib)
    nengo.Connection(inhibit, error.neurons, transform=[[-1]] * error.n_neurons, 
                                                synapse=0.01)
    
    
sim = nengo.Simulator(model)
sim.run(5)