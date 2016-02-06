##
## Bayesian HMM
##
import os
import sys
import time

import numpy as np
import particlefever
import particlefever.math_utils as math_utils
import particlefever.markov_utils as markov_utils

class DiscreteBayesHMM:
    """
    Discrete Bayesian HMM.
    """
    def __init__(self, init_probs, trans_mat, out_mat,
                 num_hidden_states,
                 num_outputs,
                 init_state_hyperparams=None,
                 trans_mat_hyperparams=None,
                 out_mat_hyperparams=None):
        self.model_type = "discrete_bayes_hmm"
        self.num_hidden_states = num_hidden_States
        self.trans_mat = trans_mat
        self.out_mat = out_mat
        # hyperparameters for prior on initial state
        self.default_init_state_hyperparam = 1
        self.init_state_hyperparams = init_state_hyperparams
        if self.init_state_hyperparams is None:
            self.init_state_hyperparam = \
              np.ones((num_states,
                       num_outputs)) * self.default_init_state_hyperparam
        # hyperparameters for prior on transition matrix
        self.trans_mat_hyperparams = trans_mat_hyperparams
        # hypeperameters for prior on output matrix
        self.out_mat_hyperparams = out_mat_hyperparams
        # default prior hyperparameters for transition and output
        # matrix
        self.default_trans_mat_hyperparam = 0.8
        self.default_out_mat_hyperparam = 1
        # initialize priors if none given
        self.trans_mat_hyperparams = trans_mat_hyperparams
        if self.trans_mat_hyperparams is None:
            self.trans_mat_hyperparams = np.ones((num_states, num_states))
            self.trans_mat_hyperparams = \
              self.trans_hyperprams * self.default_trans_mat_hyperparam
        self.out_mat_hyperparams = out_mat_hyperparams
        if self.out_mat_hyperparams is None:
            self.out_mat_hyperparams = np.ones((num_states, num_outputs))
            self.out_mat_hyperparams = \
              self.out_mat_hyperparams * self.default_out_mat_hyperparam
        # HMM state
        # hidden state assignments
        self.hidden_trajectory = np.zeros((self.num_hidden_states,
                                           self.num_hidden_states))
        self.init_state = 0
        self.hidden_state_trajectory = 0
        self.out_state_trajectory = 0
        self.trans_mat = np.zeros((self.num_hidden_states,
                                   self.num_hidden_states))
        self.out_mat = np.zeros((self.num_outputs,
                                 self.num_outputs))

    def initialize(self):
        """
        initialize to random model.
        """
        # choose initial state
        self.init_state = self.sample_init_state(self.init_state_hyperparams)
        # choose transition matrix
        self.trans_mat = init_trans_mat(self.trans_mat_hyperparams)
        # choose observation matrix
        self.out_mat = init_out_mat(self.out_mat_hyperparams)


##
## initialization functions
##
def init_trans_mat(trans_mat_hyperparams):
    """
    Sample initial transition matrix.
    """
    trans_mat = np.zeros((trans_mat_hyperparams.shape[1],
                          trans_mat_hyperparams.shape[1]))
    for n in xrange(trans_mat_hyperparams.shape[1]):
        trans_mat[n, :] = np.random.dirichlet(trans_mat_hyperparams[n, :])
    return trans_mat

def init_out_mat(out_mat_hyperparams):
    out_mat = np.zeros((out_mat_hyperparams.shape[1],
                        out_mat_hyperparams.shape[1]))
    for n in xrange(out_mat_hyperparams.shape[1]):
        out_mat[n, :] = np.random.dirichlet(out_mat_hyperparams[n, :])
    return out_mat
        
##    
## sampling functions
##
def sample_init_state(init_state_probs):
    """
    Sample assignment of initial state prior given initial state
    hyperparameters.
    """
    return np.random.multinomial(init_state_probs)

def sample_init_state_prior(init_state, init_state_prior_hyperparams):
    """
    Assume S_0 is the Dirichlet parameter that determines
    value of the initial state S_1. Sample from its
    conditional distribution:

    P(S_0 | S_1) \propto P(S_1 | S_0)P(S_0)
    """
    init_hyperparams = init_state_prior_hyperparams.copy()
    init_hyperparams[init_state] += 1
    return np.random.dirichlet(init_hyperparams)

def sample_next_hidden_state(prev_hidden_state,
                             trans_mat):
    """
    Sample next hidden state given transition matrix.

    P(S_t | S_t-1, T)
    """
    next_hidden_state = \
      np.random.multinomial(1, trans_mat[prev_hidden_state, :])
    return next_hidden_state

def sample_out_state(hidden_state, out_mat):
    out_state = np.random.multinomial(1, out_mat[hidden_state, :])
    return out_state

def sample_trans_mat(hidden_state_trajectory,
                     trans_hyperparams):
    """
    Sample transition matrix.
    """
    # here compute the matrix transition counts (using
    # bincount.)
    trans_mat_shape = (trans_hyperparams.shape[0],
                       trans_hyperparams.shape[0])
    counts_mat = \
      markov_utils.count_trans_mat(hidden_state_trajectory,
                                   trans_mat_shape)
    # sample new transition matrix by iterating through
    # rows
    sampled_trans_mat = np.zeros(trans_mat_shape)
    for n in xrange(sampled_trans_mat.shape[0]):
        # add counts from likelihood (counts matrix) and
        # from prior hyperparameters
        trans_row_params = counts_mat[n, :] + trans_hyperparams[n, :]
        sampled_trans_mat[n, :] = np.random.dirichlet(trans_row_params)
    return sampled_trans_mat

def sample_out_mat(out_trajectory, hidden_state_trajectory,
                   out_hyperparams,
                   num_hidden_states):
    """
    Sample output (observation) matrix.
    """
    num_outputs = out_hyperparams.shape[1]
    sampled_out_mat = np.zeros((num_hidden_states, num_outputs))
    for n in xrange(num_hidden_states):
        # number of occurences of each output state
        out_counts = np.bincount(out_trajectory, minlength=num_outputs)
        print "out counts: ", out_counts
        sampled_out_mat[n, :] = \
          np.random.dirichlet(out_counts + out_hyperparams[n, :])
    return sampled_out_mat

##
## scoring functions
##
def log_score_joint():
    """
    Score full joint model.
    """
    pass


def log_score_hidden_state_trajectory(hidden_trajectory,
                                      observations,
                                      trans_mat,
                                      out_mat,
                                      init_probs):
    """
    Score hidden state assignment of a single node, given observations,
    transition matrix, output matrix, and initial state probabilities.
    Return a vector of log scores:

    log[P(hidden_trajectory | observations, trans_mat, obs_mat, init_probs])
    """
    num_obs = observations.shape[0]
    log_scores = np.zeros(num_obs, dtype=np.float64)
    # handle special case of single observation
    # the score is the probability of being
    # in this initial state and emitting the single observation
    init_hidden_state = hidden_trajectory[0]
    # initial observation
    init_out = observations[0]
    if num_obs == 1:
        # prob. of being in this initial state plus
        # times prob. of observing given output
        log_scores[0] = \
          np.log(init_probs[init_hidden_state]) + \
          np.log(out_mat[init_hidden_state, init_out])
        return log_scores
    # score initial state: it depends on its output and
    # the next hidden state.
    # prob. of being in initial hidden state times 
    # prob. of observing output given initial state
    log_scores[0] = \
      np.log(init_probs[init_hidden_state]) + \
      np.log(out_mat[init_hidden_state, init_out])
    ### vectorized version
    log_scores[1:] = \
      np.log(trans_mat[hidden_trajectory[0:-1],
                       hidden_trajectory[1:]]) + \
      np.log(out_mat[hidden_trajectory[1:],
                     observations[1:]])
    return log_scores

if __name__ == "__main__":
    trans_mat = np.matrix([[0.9, 0.1],
                           [0.1, 0.9]])
    out_mat = np.matrix([0.5, 0.5])
    #DiscreteBayesHMM()
