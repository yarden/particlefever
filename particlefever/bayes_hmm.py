##
## Bayesian HMM
##
import os
import sys
import time

import numpy as np
import particlefever
import particlefever.math_utils as math_utils

class DiscreteBayesHMM:
    """
    Discrete Bayesian HMM.
    """
    def __init__(self, init_probs, trans_mat, out_mat,
                 prior_trans_mat=None,
                 prior_out_mat=None):
        self.model_type = "discrete_bayes_hmm"
        self.trans_mat = trans_mat
        self.out_mat = out_mat
        # init probs should be an array
        self.init_probs = init_probs
        assert (type(self.init_probs) == np.ndarray), \
          "init probabilities should be an np.array."
        # init probs should be 1-dimensional
        assert (self.init_probs.ndim == 1), \
          "init probabilities should be 1-dim np.array."
        self.prior_trans_mat = prior_trans_mat
        self.prior_out_mat = prior_out_mat
        self.num_hidden_nodes = trans_mat.shape[1]
        # hidden state assignments
        self.hidden_trajectory = np.zeros(self.num_hidden_nodes)
        # default hyperparameter for prior on 
        # transition matrix
        self._prior_trans_mat_dirch = 0.9
        # default hyperparameter for prior on emission
        # matrix
        self._prior_trans_mat_dirch = 1
        # initialize priors if none given
        if self.prior_trans_mat is None:
            self.prior_trans_mat = self._prior_trans_mat_dirch
        if self.prior_out_mat is None:
            self.prior_out_mat = self._prior_trans_out_mat_dirch

    def initialize(self):
        """
        initialize to random model.
        """
        # choose initial state
        self.hidden_trajectory[0] = np.random.multinomial(1, self.init_probs)
        # choose transition and emission matrices
        self.trans_mat = None

        
##    
## sampling functions
##
def sample_init_state(init_state_hyperparams):
    """
    Sample assignment of initial state prior given initial state
    hyperparameters.
    """
    return np.random.multinomial(init_state_hyperparams)

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
    sampled_trans_mat = np.zeros((trans_hyperparams.shape[0],
                                  trans_hyperparams.shape[1]))
    for n in xrange(sampled_trans_mat.shape[0]):
        # add counts from likelihood (counts matrix) and
        # from hyperparameters
        trans_row_params = counts_mat[n, :] + trans_hyperparams[n, :]
        sampled_trans_mat[n, :] = np.random.dirichlet(trans_row_params)
    return sampled_trans_mat
    

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
    # for n in xrange(1, num_obs):
    #     # the score of a state on the previous state,
    #     # the state's output, and the next state
    #     prev_hidden_state = hidden_state_trajectory[n - 1]
    #     curr_hidden_state = hidden_state_trajectory[n]
    #     out_state = observations[curr_hidden_state]
    #     # score transition from previous to current state:
    #     # P(curr_state | prev_state, trans_mat)
    #     log_score_prev_trans = trans_mat[prev_hidden_state, curr_hidden_state]
    #     # score the current output:
    #     # P(curr_out | curr_state)
    #     log_score_output = np.log(out_mat[curr_hidden_state])
    #     # final log score
    #     log_scores[n] = log_score_prev_trans + log_score_output 
        

if __name__ == "__main__":
    trans_mat = np.matrix([[0.9, 0.1],
                           [0.1, 0.9]])
    out_mat = np.matrix([0.5, 0.5])
    DiscreteBayesHMM()
