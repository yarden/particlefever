##
## Rain-Umbrella
##
import pymc
import numpy as np

import particlefever
import particlefever.dbn as dbn

rain = pymc.Bernoulli("rain", 0.5)

def umbrella_logp(value, rain):
    """
    Umbrella node. Initial value is False.

    P(umbrella=True | rain=True)  = 0.9
    P(umbrella=True | rain=False) = 0.2
    """
    p_umb_given_rain = 0.9
    p_umb_given_no_rain = 0.2
    if rain:
        logp = pymc.bernoulli_like(value, p_umb_given_rain)
    else:
        logp = pymc.bernoulli_like(value, p_umb_given_no_rain)
    return logp


def umbrella_random(rain):
    return (np.random.rand() <= np.exp(umbrella_logp(True, rain)))



# Unobserved node
#umbrella = pymc.Stochastic(name="umbrella",
#                           doc="umbrella var",
#                           parents={"rain": rain},
#                           logp=umbrella_logp,
#                           random=umbrella_random)

# Observed node
umbrella = pymc.Stochastic(name="umbrella",
                           doc="umbrella var",
                           parents={"rain": rain},
                           logp=umbrella_logp,
                           observed=True,
                           value=True)
rain_model = pymc.Model([rain, umbrella])

# Run inference at t = 0
m = pymc.MCMC(rain_model)
m.sample(iter=5000)
print "\n"
print "rain: "
print np.mean(m.trace("rain")[:])
# Instantiate a DBN
rain_dbn = particlefever.dbn.DBN(rain_model, name="rain-umbrella-hmm")

# Specify conditional relationship
def rain_conditional_func(curr_model, prev_models):
    """
    Conditional distribution function.
    Only looks at the first prev_node (i.e. t-1).

    Takes a node and a set of previous nodes and returns
    a Distribution object (i.e. object that can be sampled).
    """
    # P(rain(t) | rain(t-1) = True) = 0.7
    p_rain_given_rain = 0.7
    # P(rain(t) | rain(t-1) = False) = 0.3
    p_rain_given_no_rain = 0.3
    
    def logp(curr_value):
        """
        Score value for rain node.
        """
        prev_rain_value = prev_models[0].rain.value
        if prev_rain_value:
            return pymc.bernoulli_like(curr_value, p_rain_given_rain)
        else:
            return pymc.bernoulli_like(curr_value, p_rain_given_no_rain)
    
    def random():
        """
        Sample value for rain node.
        """
        flip = np.random.random()
        prev_rain_value = prev_models[0].rain.value
        if prev_rain_value:
            return (flip <= p_rain_given_rain)
        else:
            return (flip <= p_rain_given_no_rain)

    return (logp, random)

# Add a time slice 
rain_dbn.add_time()
# Register conditional
rain_dbn.add_time_conditional({"rain": rain_conditional_func})
rain_dbn.set_curr_time(1)
# Forward sample
rain_dbn.forward_sample()
print "Rain DBN: "
print rain_dbn
