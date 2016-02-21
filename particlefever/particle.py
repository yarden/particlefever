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

class Particle:
    """
    Particle representation of state.
    """
    def __init__(self):
        pass


class ParticleFilter(object):
    """
    Particle filter.
    """
    def __init__(self, prior_func, trans_func, observe_func,
                 num_particles=200):
        """
        Args:
        -----
        - prior_func: prior distribution function
        - trans_func: transition distribution function
        - observe_func: observation distribution function
        """
        self.prior_func = prior_func
        self.trans_func = trans_func
        self.observe_func = observe_func
        self.num_particles = num_particles
        # particles at current time
        self.particles = []
        # previous time step particles
        self.prev_particles = []
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

    def reweight(self, data_point):
        """
        Reweight particles according to evidence, P(e | S).
        """
        for n in xrange(self.num_particles):
            self.weights[n] = self.observe_func(data_point, self.particles[n])

    def resample(self):
        """
        Resample particles by their weights.
        """
        new_particles = []
        # sample new particle indices
        new_particle_inds = sample_particle_inds(self.weights,
                                                 self.num_particles)
        for n in xrange(self.num_particles):
            new_particles.append(self.particles[n])
        # save new particles
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

##
## helper functions for particle filter
##
def sample_particle_inds(w, n):
    """
    Return n random indices, where the probability if index
    is given by w[i].
    Args:
    - w (array_like): probability weights
    - n (int):  number of indices to sample
    """
    wc = np.cumsum(w)
    # normalize
    wc /= wc[-1] 
    u = (range(n) + np.random.rand(1)) / n
    return np.searchsorted(wc, u)

def calc_Neff(w):
    """
    Calculate number of effective particles, common metric used to determine
    when to resample
    Returns:
     (float) number of effective particles
    """
    tmp = np.exp(w - np.max(w))
    tmp /= np.sum(tmp)
    return 1.0 / np.sum(np.square(tmp))


class DiscreteHMM_PF(ParticleFilter):
    """
    Particle filter for HMMs.
    """
    def __init__(self, num_hidden_states, num_outputs, num_particles=200):
        super(ParticleFilter, self).__init__(num_hidden_states,
                                             num_outputs,
                                             num_particles=num_particles)
        self.prior_func = bayes_hmm.pf_init_prior
        self.trans_func = bayes_hmm.pf_trans
        self.observe_func = bayes_hmm.pf_observe
        pf_init_prior(num_particles)

class DiscreteSwitchSSM_PF(ParticleFilter):
    """
    Particle filter for discrete switching state-space model.
    """
    pass


def main():
    pass

if __name__ == "__main__":
    main()
        
        

