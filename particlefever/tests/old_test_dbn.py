##
## Test DBN
##
import pymc

import numpy as np

# prior probability of rain
p_rain = 0.5
# make all the time steps
from collections import OrderedDict
variables = OrderedDict()
# rain observations
data = [True, True, True, True, True,
        False, False, False, False, False]
num_steps = len(data)
for n in range(num_steps):
    if n == 0:
        # Rain node at time t = 0
        rain = pymc.Bernoulli("rain_%d" %(n), p_rain)
    else:
        rain_trans = pymc.Lambda("rain_trans",
                                 lambda prev_rain=variables["rain_%d" %(n-1)]: \
                                   prev_rain*0.9 + (1-prev_rain)*0.2)
        rain = pymc.Bernoulli("rain_%d" %(n), p=rain_trans)
    umbrella_obs = pymc.Lambda("umbrella_obs",
                               lambda rain=rain: \
                                 rain*0.8 + (1-rain)*0.3)
    umbrella = pymc.Bernoulli("umbrella_%d" %(n), p=umbrella_obs,
                              observed=True,
                              value=data[n])
    variables["rain_%d" %(n)] = rain
    variables["umbrella_%d" %(n)] = umbrella

# run inference
all_vars = variables.values()
model = pymc.Model(all_vars)
m = pymc.MCMC(model)
m.sample(iter=1000)
posteriors = []
times = range(num_steps)
for t in times:
    posterior_mean = np.mean(m.trace("rain_%d" %(t))[:])
    print "Posterior prob. of rain at t = %d: %.4f" %(t, posterior_mean)
    posteriors.append(posterior_mean)

import matplotlib.pylab as plt
plt.figure()
plt.plot(times, posteriors, "-o")
plt.xlabel("Time step, t")
plt.ylabel("P(rain(t) = True | data)")
plt.show()
