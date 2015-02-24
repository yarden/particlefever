##
## Rain-Umbrella 
##
import pymc
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np

def make_model():
    @pymc.stochastic(dtype=bool)
    def rain(value=False):
        """
        Rain node. Initial value is False.
        """
        # initial rain probability is random
        return pymc.bernoulli_like(value, 0.5)

    @pymc.stochastic(dtype=bool)
    def umbrella(value=False):
        """
        Umbrella node. Initial value is False.
        """
        # initial umbrella probability is random
        return pymc.bernoulli_like(value, 0.5)
    
    return locals()


rain_hmm = pymc.Model(make_model())
print "rain hmm: ", rain_hmm
