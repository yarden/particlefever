##
## My DBN
##
import os
import sys
import time

import pymc
import numpy as np

import particlefever
import particlefever.pgm as pgm
import particlefever.dbn as dbn
import particlefever.node as node

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

##
## Test DBN creation
##
# make a PGM
rain = node.Bernoulli("rain", 0.5)
umbrella = node.Stochastic(name="umbrella",
                           parents={"rain": rain},
                           logp=umbrella_logp)
model1 = pgm.PGM([rain, umbrella])
print "model 1: "
print model1
