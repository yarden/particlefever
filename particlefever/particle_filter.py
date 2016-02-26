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
import particlefever.distributions as distributions

class ParticleFilter(object):
    """
    Particle filter.
    """
    def __init__(self, prior, trans_func, observe_func,
                 num_particles=500):
        """
        Args:
        -----
        - prior: prior object
        - trans_func: transition distribution function
        - observe_func: observation distribution function
        """
        self.prior = prior
        self.trans_func = trans_func
        self.observe_func = observe_func
        self.num_particles = num_particles
        # particles at current time
        self.particles = []
        # previous time step particles
        self.prev_particles = []
        self.weights = np.ones(num_particles) / float(self.num_particles)

    def initialize(self):
        """
        Initialize particle using prior.
        """
        self.particles, self.weights = self.prior.initialize(self.num_particles)

    def sample_trans(self, num_trans=1):
        """
        Sample transitions for particles.
        """
        for n in xrange(self.num_particles):
            self.particles[n] = self.trans_func(self.particles[n], self.prior)

    def reweight(self, data_point):
        """
        Reweight particles according to evidence, P(e | S).
        """
        norm_factor = 0.
        for n in xrange(self.num_particles):
            # weights updated multiplicatively
            self.weights[n] *= self.observe_func(data_point,
                                                 self.particles[n],
                                                 self.prior)
            norm_factor += self.weights[n]
        # renormalize weights
        self.weights /= norm_factor

    def resample(self):
        """
        Resample particles by their weights.
        """
        new_particles = []
        # sample new particle indices
        new_particle_inds = sample_particle_inds(self.weights,
                                                 self.num_particles)
        for n in xrange(self.num_particles):
            new_particles.append(self.particles[new_particle_inds[n]])
        # save new particles
        self.particles = new_particles

    def process_data(self, data):
        if len(self.particles) == 0:
            raise Exception, "Must initialize particles first."
        for n in xrange(data.shape[0]):
            # sample new transitions
            self.sample_trans(data[n])
            # correct sampled transitions based on observations
            self.reweight(data[n])
            # sample new particles
            self.resample()
        print "particles at end: "
        print "--" * 5
        for n in xrange(self.num_particles):
            print self.particles[n], " weight: ", self.weights[n]

            
class DiscreteBayesHMM_PF(ParticleFilter):
    """
    Particle filter for HMMs.
    """
    def __init__(self, num_hidden_states, num_outputs, num_particles=200):
        self.num_hidden_states = num_hidden_states
        self.num_outputs = num_outputs
        prior = bayes_hmm.ParticlePrior(num_hidden_states, num_outputs)
        super(DiscreteBayesHMM_PF, self).__init__(prior,
                                                  bayes_hmm.pf_trans_sample,
                                                  bayes_hmm.pf_observe,
                                                  num_particles=num_particles)

    def predict_output(self, num_preds):
        """
        Predict output given current set of particles.
        """
        possible_outputs = np.arange(self.num_outputs)
        output_probs = np.zeros((num_preds, self.num_outputs))
        out_mat_hyperparams = self.prior.hmm.out_mat_hyperparams
        for n in xrange(num_preds):
            # first predict transition of particles to new hidden
            # state
            new_particles = []
            for prev_particle in self.particles:
                particle = bayes_hmm.pf_trans_sample(prev_particle, self.prior)
                new_particles.append(particle)
            self.particles = new_particles
            # resample particles
            self.resample()
            # then sample an output from each particle and take
            # the average
            for p in xrange(self.num_particles):
                curr_particle = self.particles[p]
                hidden_state = curr_particle.hidden_state
                pred_dist = \
                  distributions.DirMultinomial(particle.output_counts[hidden_state, :],
                                               out_mat_hyperparams[hidden_state, :])
                sampled_output = pred_dist.sample_posterior_pred()
                self.weights[p] *= bayes_hmm.pf_observe(sampled_output,
                                                        curr_particle,
                                                        self.prior)
                output_probs[n, sampled_output] += 1
            output_probs[n, :] /= float(self.num_particles)
            print "output probs for prediction %d" %(n)
            print output_probs

    def __str__(self):
        return "DiscreteBayesHMM_PF(num_particles=%d)" %(self.num_particles)


class DiscreteSwitchSSM_PF(ParticleFilter):
    """
    Particle filter for discrete switching state-space model.
    """
    pass


##
## helper functions for particle filter
##
def sample_particle_inds(w, n):
    """
    Return n random indices, where the probability of index
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


def main():
    pass

if __name__ == "__main__":
    main()
        
        

