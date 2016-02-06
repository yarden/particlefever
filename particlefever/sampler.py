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

    def sample(self, data, num_iters=10, burn_in=100, lag=2):
        """
        Run posterior sampling.
        """
        if self.model is None:
            raise Exception, "No model to run."
        # initialize model
        old_hmm = copy.copy(self.model)
        old_hmm.initialize(data=data)
        self.samples.append(old_hmm)
        for n_iter in xrange(num_iters):
            print "old hmm: "
            print old_hmm
            new_hmm = bayes_hmm.sample_new_hmm(old_hmm, data)
            print "new hmm: "
            print new_hmm
            self.samples.append(new_hmm)
            old_hmm = new_hmm

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
                
