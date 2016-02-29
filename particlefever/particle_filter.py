##
## Particle filtering.
##
import os
import sys
import time
import copy

import numpy as np

import particlefever
import particlefever.switch_ssm as switch_ssm
import particlefever.bayes_hmm as bayes_hmm
import particlefever.distributions as distributions

from collections import OrderedDict

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

    def sample_trans(self):
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
        # reset weights to be equal
        self.weights = np.ones(self.num_particles) / self.num_particles

    def process_data(self, data):
        if len(self.particles) == 0:
            raise Exception, "Must initialize particles first."
        for n in xrange(data.shape[0]):
            # sample new transitions
            self.sample_trans()
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
        # resample particles before making predictions
        self.resample()
        for n in xrange(num_preds):
            # first predict transition of particles to new hidden
            # state
            new_particles = []
            for prev_particle in self.particles:
                particle = bayes_hmm.pf_trans_sample(prev_particle, self.prior)
                new_particles.append(particle)
            self.particles = new_particles
            # then sample an output from each particle and then
            # reweight it by the evidence
            for p in xrange(self.num_particles):
                curr_particle = self.particles[p]
                hidden_state = curr_particle.hidden_state
                pred_dist = \
                  distributions.DirMultinomial(curr_particle.output_counts[hidden_state, :],
                                               out_mat_hyperparams[hidden_state, :])
                sampled_output = pred_dist.sample_posterior_pred()
                self.weights[p] *= bayes_hmm.pf_observe(sampled_output,
                                                        curr_particle,
                                                        self.prior)
                output_probs[n, sampled_output] += self.weights[p]
            output_probs[n, :] /= self.weights.sum()
        return output_probs

    def __str__(self):
        return "DiscreteBayesHMM_PF(num_particles=%d)" %(self.num_particles)


class DiscreteSwitchSSM_PF(ParticleFilter):
    """
    Particle filter for discrete switching state-space model.
    """
    def __init__(self, num_switch_states, num_outputs, num_particles=200):
        self.num_switch_states = num_switch_states
        self.num_outputs = num_outputs
        prior = switch_ssm.ParticlePrior(num_switch_states, num_outputs)
        super(DiscreteSwitchSSM_PF, self).__init__(prior,
                                                   switch_ssm.pf_trans_sample,
                                                   switch_ssm.pf_observe,
                                                   num_particles=num_particles)

    def reweight(self, data_point, prev_output):
        """
        Reweight particles according to evidence, P(e | S).
        """
        norm_factor = 0.
        for n in xrange(self.num_particles):
            # weights updated multiplicatively
            self.weights[n] *= self.observe_func(data_point,
                                                 self.particles[n],
                                                 self.prior,
                                                 prev_output=prev_output)
            norm_factor += self.weights[n]
        # renormalize weights
        self.weights /= norm_factor

    def process_data(self, data, save_steps=True):
        """
        Filter through data (without storing particles at each time step.)
        """
        if len(self.particles) == 0:
            raise Exception, "Must initialize particles first."
        prev_output = None
        self.filter_results = OrderedDict()
        for n in xrange(data.shape[0]):
            if n > 0:
                # record previous data point if we're not
                # in the first time step
                prev_output = data[n - 1]
            # sample new transitions
            self.sample_trans()
            # correct sampled transitions based on observations
            self.reweight(data[n], prev_output=prev_output)
            # sample new particles
            self.resample()
            if save_steps:
                # save particles and their weights
                self.filter_results[n] = \
                  {"particles": copy.deepcopy(self.particles),
                   "weights": copy.deepcopy(self.weights)}
        print "particles at end: "
        print "--" * 5
        for n in xrange(self.num_particles):
            print self.particles[n], " weight: ", self.weights[n]

    def predict_output(self, num_preds, prev_output):
        """
        Predict output for number of given time steps, given
        previous output (if any).
        """
        output_probs = np.zeros((num_preds, self.num_outputs))
        out_trans_mat_hyperparams = self.prior.ssm.out_trans_mat_hyperparams
        # resample particles before making predictions
        self.resample()
        # record previous outputs for each particle
        particle_prev_outputs = np.ones(num_particles)
        if prev_output is not None:
            # at first time step, the previous output is determined by
            # the data, assuming we had observed any data
            particle_prev_outputs *= prev_output
        else:
            # special case where there's no previous output
            # in this case, draw outputs from initialized
            # particles. draw as many samples from prior as there
            # are particles
            for n in xrange(num_particles):
                prev_counts = np.zeros(self.num_outputs)
                init_out_hyperparams = self.prior.ssm.init_out_hyperparams
                pred_dist = \
                  distributions.DirMultinomial(prev_counts, init_out_hyperparams)
                sampled_output = pred_dist.sample_posterior_pred()
                particle_prev_outputs[n] = sampled_output
                output_probs[0, sampled_output] += 1
            output_probs = output_probs / float(self.num_particles)
        for n in xrange(num_preds):
            # first predict transition of particles to new switch state
            new_particles = []
            for prev_particle in self.particles:
                particle = switch_ssm.pf_trans_sample(prev_particle, self.prior)
                new_particles.append(particle)
            self.particles = new_particles
            # then sample an output from each particle and then
            # reweight it by the evidence
            for p in xrange(self.num_particles):
                prev_output = particle_prev_outputs[p]
                curr_particle = self.particles[p]
                switch_state = curr_particle.switch_state
                pred_dist = \
                  distributions.DirMultinomial(curr_particle.out_trans_counts[switch_state,
                                                                              prev_output, :],
                                               out_trans_mat_hyperparams[prev_output, :])
                sampled_output = pred_dist.sample_posterior_pred()
                self.weights[p] *= switch_ssm.pf_observe(sampled_output,
                                                         curr_particle,
                                                         self.prior,
                                                         prev_output=prev_output)
                output_probs[n, sampled_output] += self.weights[p]
                # update previous output
                particle_prev_outputs[p] = sampled_output
            output_probs[n, :] /= self.weights.sum()
        return output_probs

    def prediction_with_lag(self, lag=1):
        """
        Get prediction probabilities for a set of observations
        assuming a lag of 1 by default.
        """
        num_outputs = self.num_outputs
        num_obs = len(self.filter_results)
        if num_obs == 0:
            raise Exception, "No filtering posteriors found."
        prediction_probs = np.zeros((num_obs, num_outputs))
        print "FILTER RESULTS: "
        for k in xrange(num_obs):
            # need to add 1 here to k to get 1-based time
            # for lag computation
            if (k + 1 - lag) <= 0:
                posterior = self.filter_results[0]
            else:
                # also need to add 1 here to k to get 1-based time
                # for lag computation
                posterior = self.filter_results[k + 1 - lag]
            ###
            ### TODO: here add code to predict remaining
            ### observations using current set of particles, without
            ### adversely affecting the state of the object, i.e.
            ### self.particles or self.weights. Perhaps project forward
            ### then restore them to what they were at this time point.
            ###
            # predict remaining observations using current
            # set of particles
            num_preds = num_obs - k
            print "predicting rest of %d time points" %(num_preds)
            print "  - curr t: %d" %(k)
            self.particles = posterior["particles"]
            self.weights = posterior["weights"]
            predictions = self.predict_output(num_preds,
                                              prev_output)
            print "predictions: ", predictions
            # here don't add 1 to k; we're storing 0-based time
            prediction_probs[k, :] = predictions
        return prediction_probs

    def __str__(self):
        return "DiscreteSwitchSSM_PF(num_particles=%d)" %(self.num_particles)

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
        
        

