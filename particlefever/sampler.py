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
import particlefever.switch_ssm as switch_ssm

from collections import OrderedDict

# Node types
DISCRETE = "discrete"
CONTINUOUS = "continuous"

class GibbsSampler(object):
    def __init__(self, model):
        self.model = model
        self.samples = []
        self.sample_new_model = None
        self.filter_results = OrderedDict()

    def __str__(self):
        return "Gibbs(model=%s)" %(self.model)

    def __repr__(self):
        return self.__str__()

    def sample(self, data, num_iters=5000, burn_in=100, lag=4, verbose=False,
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
        if verbose:
            print "sampling took %.2f secs (%d final samples)" %(t2 - t1,
                                                                 num_sampled)

    def filter_fit(self, init_model, data, predict_func, **sampler_args):
        """
        Compute filtering estimate. Go through each observation k
        and from it compute probability of remaining t_total - k
        observations using prediction.

        Args:
        -----
        - init_model: initial model (prior)
        - data: data to fit
        - predict_func: prediction function, which takes a set of sampled models
          and a number of predictions, and returns an array of predictions for
          the outputs.
        """
        data = np.array(data)
        t1 = time.time()
        num_obs = data.shape[0]
        num_prior_samples = 500
        self.filter_results = OrderedDict()
        for t in xrange(num_obs):
            curr_obs = data[0:t + 1]
            if t == 0:
                # for special case of a single observation, generate
                # sample from prior
                models = init_model.sample_prior(num_obs, num_prior_samples)
            else:
                self.samples = []
                self.sample(curr_obs, **sampler_args)
                models = self.samples
            # predict rest of observations using prediction
            self.filter_results[t] = predict_func(models, num_obs - t)
        t2 = time.time()
        print "filtering fit took %.2f" %(t2 - t1)
        
    def get_prediction_probs(self, lag=1, num_outputs=2):
        """
        Get prediction probabilities for a set of observations
        assuming a lag of 1 by default.
        """
        num_obs = len(self.filter_results)
        if num_obs == 0:
            raise Exception, "No filtering posteriors found."
        prediction_probs = np.zeros((num_obs, num_outputs))
        for k in xrange(num_obs):
            # need to add 1 here to k to get 1-based time
            # for lag computation
            if (k + 1 - lag) <= 0:
                posterior = self.filter_results[0][0, :]
            else:
                # also need to add 1 here to k to get 1-based time
                # for lag computation
                posterior = self.filter_results[k + 1 - lag][0, :]
            # here don't add 1 to k; we're storing 0-based time
            prediction_probs[k, :] = posterior
        return prediction_probs

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
