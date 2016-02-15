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

class GibbsSampler(object):
    def __init__(self, model):
        self.model = model
        self.samples = []
        self.sample_new_model = None

    def __str__(self):
        return "Gibbs(model=%s)" %(self.model)

    def __repr__(self):
        return self.__str__()

    def sample(self, data, num_iters=10000, burn_in=100, lag=4,
               **kwargs):
        """
        Run posterior sampling.
        """
        if self.model is None:
            raise Exception, "No model to run."
        if self.sample_new_model is None:
            raise Exception, "Need method to set method to sample new model."
        # initialize model if not initialized
        # already
        old_model = copy.deepcopy(self.model)
        if not old_model.initialized:
            old_model.initialize()
        # add data
        old_model.add_data(data, **kwargs)
        self.samples.append(old_model)
        t1 = time.time()
        for n_iter in xrange(num_iters):
            new_model = self.sample_new_model(old_model, data)
            if (n_iter % lag == 0) and (n_iter >= burn_in):
                self.samples.append(new_model)
            old_model = new_model
        t2 = time.time()
        num_sampled = len(self.samples)
        print "sampling took %.2f secs (%d final samples)" %(t2 - t1,
                                                             num_sampled)

class DiscreteSwitchSSM(GibbsSampler):
    """
    Gibbs sampler for discrete Bayesian HMM.
    """
    def __init__(self, model):
        super(DiscreteSwitchSSM, self).__init__(model)
        self.sample_new_model = switch_ssm.sample_new_ssm

    def __str__(self):
        return "DiscreteSwitchSSMGibbs(model=%s)" %(self.model)
                             
                         
class DiscreteBayesHMMGibbs(GibbsSampler):
    """
    Gibbs sampler for discrete Bayesian HMM.
    """
    def __init__(self, model):
        super(DiscreteBayesHMMGibbs, self).__init__(model)
        self.sample_new_model = bayes_hmm.sample_new_hmm

    def __str__(self):
        return "DiscreteBayesHMMGibbs(model=%s)" %(self.model)
