##
## Particle filtering.
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
                 state_space_dim,
                 num_particles=200):
        """
        Args:
        -----
        - prior_func: prior distribution function
        - trans_func: transition distribution function
        - observe_func: observation distribution function
        - state_space_dim: dimensionality of state space
          to be represented by particle
        """
        self.prior_func = prior_func
        self.trans_func = trans_func
        self.observe_func = observe_func
        self.num_particles = num_particles
        self.particles = np.array((num_particles, state_space_dim))
        self.weights = np.array(num_particles)

    def init(self):
        """
        Initialize particle using prior.
        """
        for n in xrange(self.num_particles):
            self.particles[n] = self.prior_func()

    def sample_trans(self, num_trans=1):
        """
        Sample transitions for particles.
        """
        for n in xrange(self.num_particles):
            self.particles[n] = self.trans_func(self.particles[n])

    def reweight(self):
        """
        Reweight particles according to evidence.
        """
        

    def resample(self):
        """
        Resample particles by their weights.
        """
        new_particles = np.array((num_particles, state_space_dim))
        for n in xrange(self.num_particles):
            particle_ind = np.random.multinomial(1, self.weights).argmax()
            new_particles[n] = self.particles[n]
        self.particles = new_particles

    def next_time(self, data_point):
        """
        Process next time point.
        """
        # sample new transitions
        self.sample_trans(data_point)
        # correct sampled transitions based on observations
        self.reweight()
        # sample new particles
        self.resample()

class DiscreteHMM_PF(ParticleFilter):
    """
    Particle filter for HMMs.
    """
    pass

class DiscreteSwitchSSM_PF(ParticleFilter):
    """
    Particle filter for discrete switching state-space model.
    """
    pass


def main():
    pass

if __name__ == "__main__":
    main()
        
        

