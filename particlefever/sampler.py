##
## Simple sampler
##
import os
import sys
import time

import copy

import particlefever
import particlefever.math_utils as math_utils

import numpy as np

# Node types
DISCRETE = "discrete"
CONTINUOUS = "continuous"

class Sampler(object):
    def __init__(self, model, num_iters=100, burn_in=10, lag=2):
        # model to run on
        self.model = model
        # copy of model that can be modified through
        # iteration
        self.sampler_model = copy.deepcopy(self.model)
        self.num_iters = num_iters
        self.burn_in = burn_in
        self.lag = lag

    def gibbs_sample_discrete_node(self, node):
        """
        Sample value for discrete node.
        """
        # Evaluate log score for each value of node
        # given all other nodes
        possible_values = node.get_possible_values()
        log_joint_scores = []
        for value in possible_values:
            # Calculate log joint score of model with
            # node taking on current value
            self.sampler_model.node.set_value(value)
            joint_log_score = self.sampler_model.get_joint_log_score()
            log_joint_scores.append(joint_log_score)
        # Multinomial sample it
        log_joint_scores = np.array(log_joint_scores)
        math_utils.sample_multinomial_logprobs(log_joint_scores)
        return value

    def run_gibbs(self):
        # First initialize values to all nodes
        self.sampler_model.initialize_values()
        # Run Gibbs sampling
        for curr_iter in self.num_iters:
            for node in self.sampler_model:
                # sample new value for current node conditioned
                # on values for all other nodes
                if node.data_type == DISCRETE:
                    self.gibbs_sample_discrete_node(node)
                else:
                    raise Exception, "Only support discrete nodes."
            
    def run_mh(self):
        pass


