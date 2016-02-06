##
## Bayesian HMM
##
import os
import sys
import time
import copy

import numpy as np
import particlefever
import particlefever.math_utils as math_utils
import particlefever.markov_utils as markov_utils

##
## TODO:
##
## Write log-based multinomial probabilities sampling function
## put in prob_utils.py
##
##  - Add unit test for small probabilities
##

class DiscreteBayesHMM:
    """
    Discrete Bayesian HMM.
    """
    def __init__(self, num_hidden_states, num_outputs,
                 init_state_hyperparams=None,
                 trans_mat_hyperparams=None,
                 out_mat_hyperparams=None):
        self.model_type = "discrete_bayes_hmm"
        # HMM state
        self.num_hidden_states = num_hidden_states
        self.num_outputs = num_outputs
        # hidden state assignments
        self.hidden_trajectory = np.zeros((self.num_hidden_states,
                                           self.num_hidden_states))
        self.hidden_state_trajectory = 0
        self.out_state_trajectory = 0
        self.trans_mat = np.zeros((self.num_hidden_states,
                                   self.num_hidden_states))
        self.out_mat = np.zeros((self.num_outputs,
                                 self.num_outputs))
        # hyperparameters for prior on initial state
        self.default_init_state_hyperparam = 1
        self.init_state_hyperparams = init_state_hyperparams
        if self.init_state_hyperparams is None:
            self.init_state_hyperparams = np.ones(self.num_outputs)
            self.init_state_hyperparams *= self.default_init_state_hyperparam
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
            self.trans_mat_hyperparams = \
              np.ones((num_hidden_states, num_hidden_states))
            self.trans_mat_hyperparams *= self.default_trans_mat_hyperparam
        self.out_mat_hyperparams = out_mat_hyperparams
        if self.out_mat_hyperparams is None:
            self.out_mat_hyperparams = np.ones((num_hidden_states,
                                                num_outputs))
            self.out_mat_hyperparams *= self.default_out_mat_hyperparam

    def __str__(self):
        return "DiscreteBayesHMM(num_hidden_states=%d, num_outputs=%d,\n" \
               "trans_mat=%s,\nout_mat=%s)" \
               %(self.num_hidden_states,
                 self.num_outputs,
                 self.trans_mat,
                 self.out_mat)

    def __repr__(self):
        return self.__str__()

    def initialize(self, data=None):
        """
        initialize to random model.
        """
        # choose initial state probabilities
        self.init_state_probs = np.random.dirichlet(self.init_state_hyperparams)
        # choose transition matrix
        self.trans_mat = init_trans_mat(self.trans_mat_hyperparams)
        # choose observation matrix
        self.out_mat = init_out_mat(self.out_mat_hyperparams)
        # if given data, initialize hidden state trajectory
        if data is not None:
            self.data_len = data.shape[0]
            self.hidden_state_trajectory = np.zeros(self.data_len,
                                                    dtype=np.int32)
            # choose initial state
            self.hidden_state_trajectory[0] = \
              np.random.multinomial(1, self.init_state_probs).argmax()
            # choose hidden state trajectories
            for n in xrange(1, self.data_len):
                prev_hidden_state = self.hidden_state_trajectory[n - 1]
                self.hidden_state_trajectory[n] = \
                  np.random.multinomial(1, self.trans_mat[prev_hidden_state, :]).argmax()
            # choose outputs
            self.outputs = np.zeros(self.data_len, dtype=np.int32)
            for n in xrange(self.data_len):
                out_probs = self.out_mat[self.hidden_state_trajectory[n], :]
                self.outputs[n] = np.random.multinomial(1, out_probs).argmax()
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
    out_mat = np.zeros((out_mat_hyperparams.shape[0],
                        out_mat_hyperparams.shape[1]))
    for n in xrange(out_mat_hyperparams.shape[0]):
        out_mat[n, :] = np.random.dirichlet(out_mat_hyperparams[n, :])
    return out_mat
        
##    
## sampling functions
##
def sample_new_hmm(old_hmm, data):
    new_hmm = copy.copy(old_hmm)
    # sample initial state probs
    new_hmm.init_state_probs = \
      sample_init_state_prior(new_hmm.hidden_state_trajectory[0],
                              new_hmm.init_state_hyperparams)
    # sample initial state
    new_hmm.init_state = sample_init_state(new_hmm.init_state_probs)
    # sample transition matrix
    new_hmm.trans_mat = sample_trans_mat(new_hmm.hidden_state_trajectory,
                                         new_hmm.trans_mat_hyperparams)
    # sample hidden states
    new_hmm.hidden_state_trajectory = \
      sample_hidden_states(new_hmm.hidden_state_trajectory,
                           new_hmm.trans_mat,
                           new_hmm.init_state_probs)
    # sample output matrix
    new_hmm.out_mat = sample_out_mat(new_hmm.outputs,
                                     new_hmm.hidden_state_trajectory,
                                     new_hmm.out_mat_hyperparams,
                                     new_hmm.num_hidden_states)
    # sample outputs
    new_hmm.outputs = sample_outputs(new_hmm.hidden_state_trajectory,
                                     new_hmm.out_mat)
    return new_hmm
    

def sample_init_state_prior(init_state, init_state_hyperparams):
    """
    Assume S_0 is the Dirichlet parameter that determines
    value of the initial state S_1. Sample from its
    conditional distribution:

    P(S_0 | S_1) \propto P(S_1 | S_0)P(S_0)
    """
    init_hyperparams = init_state_hyperparams.copy()
    init_hyperparams[init_state] += 1
    return np.random.dirichlet(init_hyperparams)
    
def sample_init_state(init_state_probs):
    """
    Sample assignment of initial state prior given initial state
    hyperparameters.
    """
    return np.random.multinomial(1, init_state_probs).argmax()

def sample_hidden_states(hidden_state_trajectory,
                         trans_mat,
                         init_state_probs):
    """
    Sample new configuration of hidden states.
    """
    num_hidden_states = trans_mat.shape[0]
    new_hidden_state_trajectory = np.zeros((num_hidden_states,
                                            num_hidden_states),
                                           dtype=np.int32)
    # sample initial state
    hidden_state_trajectory[0] = sample_init_state(init_state_probs)
    for n in xrange(1, num_hidden_states):
        prev_hidden_state = hidden_state_trajectory[n - 1]
        hidden_state_trajectory[n] = \
          sample_next_hidden_state(prev_hidden_state, trans_mat)
    return hidden_state_trajectory

def sample_next_hidden_state(prev_hidden_state,
                             trans_mat):
    """
    Sample next hidden state given transition matrix.

    P(S_t | S_t-1, T)
    """
    next_hidden_state = \
      np.random.multinomial(1, trans_mat[prev_hidden_state, :]).argmax()
    return next_hidden_state

def sample_out_state(hidden_state, out_mat):
    out_state = np.random.multinomial(1, out_mat[hidden_state, :]).argmax()
    return out_state

def sample_trans_mat(hidden_state_trajectory,
                     trans_mat_hyperparams):
    """
    Sample transition matrix.
    """
    # here compute the matrix transition counts (using
    # bincount.)
    trans_mat_shape = (trans_mat_hyperparams.shape[0],
                       trans_mat_hyperparams.shape[0])
    counts_mat = \
      markov_utils.count_trans_mat(hidden_state_trajectory,
                                   trans_mat_shape)
    # sample new transition matrix by iterating through
    # rows
    sampled_trans_mat = np.zeros(trans_mat_shape)
    for n in xrange(sampled_trans_mat.shape[0]):
        # add counts from likelihood (counts matrix) and
        # from prior hyperparameters
        trans_row_params = counts_mat[n, :] + trans_mat_hyperparams[n, :]
        sampled_trans_mat[n, :] = np.random.dirichlet(trans_row_params)
    return sampled_trans_mat

def sample_out_mat(outputs,
                   hidden_state_trajectory,
                   out_mat_hyperparams,
                   num_hidden_states):
    """
    Sample output (observation) matrix.
    """
    num_outputs = out_mat_hyperparams.shape[1]
    sampled_out_mat = np.zeros((num_hidden_states, num_outputs))
    for n in xrange(num_hidden_states):
        # number of occurences of each output state
        out_counts = np.bincount(outputs, minlength=num_outputs)
        sampled_out_mat[n, :] = \
          np.random.dirichlet(out_counts + out_mat_hyperparams[n, :])
    return sampled_out_mat

def sample_outputs(hidden_state_trajectory, out_mat):
    """
    Sample outputs.
    """
    outputs = np.zeros(hidden_state_trajectory.shape[0],
                       dtype=np.int32)
    for n in xrange(hidden_state_trajectory.shape[0]):
        hidden_state = hidden_state_trajectory[n]
        outputs[n] = np.random.multinomial(1, out_mat[hidden_state, :]).argmax()
    return outputs

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
