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
        self.trans_mat
        pass

##    
## scoring functions
##
def log_score_hidden_state_trajectory(hidden_trajectory,
                                      observations,
                                      trans_mat,
                                      out_mat,
                                      init_probs):
    """
    Score hidden state assignment of a single node, given observations,
    transition matrix, output matrix, and initial state probabilities.
    Return a vector of log scores:

    log[P(hidden_trajectory | observations, trans_mat, obs_mat, init_probs)]
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
