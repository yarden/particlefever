##
## Particle filtering
##
import os
import sys
import time

import numpy as np

import particlefever
import particlefever.switch_ssm as switch_ssm
import particlefever.bayes_hmm as bayes_hmm

class ParticleFilter(object):
    """
    Particle filter.
    """
    def __init__(self, prior_func, trans_func, observe_func,
                 num_particles=200):
        """
        Args:
        -----
        - prior_dist: prior distribution function
        - trans_func: transition distribution function
        - observe_func: observation distribution function
        """
        self.prior_func = prior_func
        self.trans_func = trans_func
        self.observe_func = observe_func
        self.num_particles = num_particles

    def sample_transitions(self):
        pass

class DiscreteHMMParticleFilter(ParticleFilter):
    pass


def main():
    pass

if __name__ == "__main__":
    main()
        
        

