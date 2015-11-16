##
## Test DBN
##
import os
import sys
import time

import particlefever
import particlefever.pgm as pgm
import particlefever.dbn as dbn
import particlefever.node as node

import pymc3
import numpy as np

num_steps = 10
from collections import OrderedDict
variables = OrderedDict()
model = pymc3.Model()


class Rain(pymc3.distributions.Discrete):
    def __init__(self, *args, **kwargs):
        super(Rain, self).__init__(*args, **kwargs)

    def logp(self, value, prev_rain):
        prev_rain_val = prev_rain.random()
        if prev_rain_val:
            return value*np.log(0.9) + value*np.log(0.1)
        else:
            return value*np.log(0.3) + value*np.log(0.7)

with model:
    for n in range(num_steps):
        if n == 0:
            # Rain node at time t = 0 
            rain = pymc3.Bernoulli("rain_%d" %(n), 0.5)
        else:
            def rain_trans(prev_rain):
                # sample value for previous rain node with 'random',
                # (since .value not available for RVs in PyMC3)
                if prev_rain.random():
                    print "previous rain was true"
                    return 0.9
                else:
                    print "previous rain was false"
                    return 0.3
            #rain = pymc3.Bernoulli("rain_%d" %(n), rain_trans(variables["rain_%d" %(n-1)]))
            rain = Rain("p", "rain_%d" %(n), variables["rain_%d" %(n-1)])
        # define umbrella node
        # def umbrella_cpt(rain_node):
        #     if rain_node.random():
        #         return 0.8
        #     else:
        #         return 0.3
        # umbrella = pymc3.Bernoulli("umbrella_%d" %(n), umbrella_cpt(rain),
        #                            observed=True)
        # variables["umbrella_%d" %(n)] = umbrella
        # save variable nodes
        variables["rain_%d" %(n)] = rain
    start = pymc3.find_MAP()
    step = pymc3.Metropolis()
    trace = pymc3.sample(100, step, start, random_seed=123, progressbar=True)
    print "Mean value of rain @ t=8"
    print np.mean(trace("rain_8")[:])


print "-" * 20

# Now specify that umbrella depends on rain.
# It seems that I have to declare that umbrella
# is observed, as well as its observed value, here -- as part
# of model declaration.  Unlike Church, Pymc3 doesn't
# separate model definition from observations?
# umbrella = pymc3.Stochastic(logp=umbrella_logp,
#                            name="umbrella_%d" %(n),
#                            doc="umbrella %d" %(n),
#                            parents={"rain": rain},
#                            random=umbrella_random,
#                            observed=True,
#                            value=True)
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
