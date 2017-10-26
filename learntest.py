#Setup the environment

import numpy as np
import nengo
from nengo.processes import WhiteSignal

model = nengo.Network(label='Learning', seed=8)
with model:
    #Ensembles to represent populations
    pre = nengo.Ensemble(50, dimensions=1)
    post = nengo.Ensemble(50, dimensions=1)
    error = nengo.Ensemble(100, dimensions=1)
    actual_error = nengo.Ensemble(100, dimensions=1, neuron_type=nengo.Direct())
    
    #Actual Error = pre - post (direct mode)
    #Square
    #nengo.Connection(pre, actual_error, function=lambda x: x**2, transform=-1)   
    nengo.Connection(pre, actual_error, transform=-1)   #Communication Channel
    nengo.Connection(post, actual_error, transform=1)
     
    #Error = pre - post
    #Square
    #nengo.Connection(pre, error, function=lambda x: x**2, transform=-1)
    #Communication Channel
    nengo.Connection(pre, error, transform=-1, synapse=0.02)
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
    input = nengo.Node(WhiteSignal(30, high=10))     # RMS = 0.5 by default
    # Connecting input to the pre ensemble
    nengo.Connection(input, pre, synapse=0.02)  
    
    #function to inhibit the error population after 25 seconds
    def inhib(t):
        return 2.0 if t > 15.0 else 0.0
    
    #Connecting inhibit population to error population
    inhibit = nengo.Node(inhib)
    nengo.Connection(inhibit, error.neurons, transform=[[-1]] * error.n_neurons, 
                                                synapse=0.01)
                                                
                                                