##
## Samplers
##
import os
import sys
import time

import copy
import numpy as np

import particlefever
import particlefever.math_utils as math_utils
import particlefever.bayes_hmm as bayes_hmm

# Node types
DISCRETE = "discrete"
CONTINUOUS = "continuous"

class Sampler(object):
    def __init__(self):
        pass

class DiscreteBayesHMMGibbs(Sampler):
    """
    Gibbs sampler for discrete Bayesian HMM.
    """
    def __init__(self, model):
        self.model = model
        self.samples = []

    def __str__(self):
        return "DiscreteBayesHMMGibbs(model=%s)" %(self.model)

    def __repr__(self):
        return self.__str__()

    def sample(self, data, num_iters=10000, burn_in=100, lag=4,
               init_hidden_states=True):
        """
        Run posterior sampling.
        """
        if self.model is None:
            raise Exception, "No model to run."
        # initialize model if not initialized
        # already
        old_hmm = copy.deepcopy(self.model)
        if old_hmm.hidden_state_trajectory is None:
            old_hmm.initialize()
        # add data
        old_hmm.add_data(data, init_hidden_states=init_hidden_states)
        self.samples.append(old_hmm)
        t1 = time.time()
        for n_iter in xrange(num_iters):
            new_hmm = bayes_hmm.sample_new_hmm(old_hmm, data)
            if (n_iter % lag == 0) and (n_iter >= burn_in):
                self.samples.append(new_hmm)
            old_hmm = new_hmm
        t2 = time.time()
        num_sampled = len(self.samples)
        print "sampling took %.2f secs (%d final samples)" %(t2 - t1,
                                                             num_sampled)

    def summarize_outputs(self):
        """
        Summarize the outputs.
        """
        pass

    def save_outputs(self):
        """
        Save output as pickle.
        """
        pass
                
