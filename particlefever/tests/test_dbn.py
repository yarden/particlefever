##
## Test DBN
##
import os
import sys
import time

import pymc

import copy

import numpy as np

import particlefever
import particlefever.pgm as pgm
import particlefever.dbn as dbn
import particlefever.node as node

import pymc

def umbrella_logp(value, rain):
    """
    Umbrella node.

    P(umbrella=True | rain=True)  = 0.9
    P(umbrella=True | rain=False) = 0.2
    """
    p_umb_given_rain = 0.9
    p_umb_given_no_rain = 0.2
    if rain:
        logp = pymc.bernoulli_like(rain, p_umb_given_rain)
    else:
        logp = pymc.bernoulli_like(rain, p_umb_given_no_rain)
    return logp

def umbrella_random(rain):
    return (np.random.rand() <= np.exp(umbrella_logp(True, rain)))

def rain_logp(value, rain):
    """
    P(rain=True @ t | rain=True @ t-1) = 0.8
    p(rain=True @ t | rain=False @ t-1) = 0.35
    """
    if rain:
        logp = pymc.bernoulli_like(rain, 0.8)
    else:
        logp = pymc.bernoulli_like(rain, 0.35)
    return logp

def rain_random(rain):
    return (np.random.rand() <= np.exp(rain_logp(True, rain)))


num_steps = 10
# prior probability of rain
p_rain = 0.5
# make all the time steps
from collections import OrderedDict
variables = OrderedDict()
for n in range(num_steps):
    if n == 0:
        # Rain node at time t = 0
        rain = pymc.Bernoulli("rain_%d" %(n), p_rain)
    else:
        # Encode probability of rain given previous day's rain
        # This fails because rain_logp doesn't know in advance
        # the argument name
        rain = pymc.Stochastic(logp=rain_logp,
                               name="rain_%d" %(n),
                               doc="rain_%d" %(n),
                               parents={"rain":
                                        variables["rain_%d" %(n-1)]},
                               random=rain_random)
    # Now specify that umbrella depends on rain.
    # It seems that I have to declare that umbrella
    # is observed, as well as its observed value, here -- as part
    # of model declaration.  Unlike Church, PyMC doesn't
    # separate model definition from observations?
    umbrella = pymc.Stochastic(logp=umbrella_logp,
                               name="umbrella_%d" %(n),
                               doc="umbrella %d" %(n),
                               parents={"rain": rain},
                               random=umbrella_random,
                               observed=True,
                               value=True)
    variables["rain_%d" %(n)] = rain
    variables["umbrella_%d" %(n)] = umbrella

# run inference
all_vars = variables.values()
m = pymc.MCMC(all_vars)
m.sample(iter=1000)

print "Mean value of rain @ t=8"
print np.mean(m.trace("rain_8")[:])


print "-" * 20


##
## Test DBN creation
##
# make a PGM
#rain = node.Bernoulli("rain", 0.5)

#umbrella = node.Stochastic(name="umbrella",
#                           parents={"rain": rain},
#                           logp=umbrella_logp)
#model1 = pgm.PGM([rain, umbrella])
#print "model 1: "
#print model1
