##
## Rain-Umbrella 
##
import pymc
import numpy as np

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


umbrella = pymc.Stochastic(name="umbrella",
                           doc="umbrella var",
                           parents={"rain": rain},
                           logp=umbrella_logp,
                           observed=True,
                           value=True)
rain_hmm = pymc.Model([rain, umbrella])

if __name__ == "__main__":
    # Run inference
    m = pymc.MCMC(rain_hmm)
    m.sample(iter=5000)
    print "\n"
    print "rain: "
    print np.mean(m.trace("rain")[:])


