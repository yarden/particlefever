##
## Switching state space model
##
import os
import sys
import time
import copy

import numpy as np

import particlefever
import particlefever.math_utils as math_utils
import particlefever.markov_utils as markov_utils

import scipy
import scipy.stats

DEBUG = True
DEBUG_DELAY = 0.5

class DiscreteSwitchSSM:
    """
    Discrete switching state-space model.
    """
    def __init__(self, num_switch_states, num_outputs,
                 init_switch_hyperparams=None,
                 init_out_hyperparams=None,
                 switch_trans_mat_hyperparams=None,
                 out_trans_mat_hyperparams=None):
        self.model_type = "discrete_switch_ssm"
        self.initialized = False
        self.num_switch_states = num_switch_states
        self.num_outputs = num_outputs
        # switch state assignments
        self.switch_state_trajectory = None
        # switch state transition matrices
        self.state = None
        # output transition matrices: one for each switch state
        self.out_trans_mats = np.zeros((self.num_switch_states,
                                        self.num_outputs,
                                        self.num_outputs))
        # hyperparameters for prior on initial output
        self.init_out_hyperparams = init_out_hyperparams
        self.default_init_out_hyperparams = 1.
        if self.init_out_hyperparams is None:
            self.init_out_hyperparams = np.ones(self.num_outputs)
            self.init_out_hyperparams *= self.default_init_out_hyperparams
        # hyperparameters for prior on initial switch state
        self.default_init_switch_hyperparam = 1.
        self.init_switch_hyperparams = init_switch_hyperparams
        if self.init_switch_hyperparams is None:
            self.init_switch_hyperparams = np.ones(self.num_switch_states)
            self.init_switch_hyperparams *= self.default_init_switch_hyperparam
        # hyperparameters for prior on switch state transition matrix
        self.switch_trans_mat_hyperparams = switch_trans_mat_hyperparams
        # hypeperameters for prior on output transition matrix
        self.out_trans_mat_hyperparams = out_trans_mat_hyperparams
        # default prior hyperparameters for transition and output
        # matrix
        self.default_trans_mat_hyperparam = 0.8
        self.default_out_trans_mat_hyperparam = 1.
        # make default switch transition matrix
        if self.switch_trans_mat_hyperparams is None:
            self.switch_trans_mat_hyperparams = np.ones((self.num_switch_states,
                                                     self.num_switch_states))
            self.switch_trans_mat_hyperparams *= self.default_trans_mat_hyperparam
        # make default output transition matrix
        if self.out_trans_mat_hyperparams is None:
            self.out_trans_mat_hyperparams = np.ones((self.num_outputs,
                                                      self.num_outputs))
            self.out_trans_mat_hyperparams *= self.default_out_trans_mat_hyperparam

    def __str__(self):
        return "DiscreteSwitchSSM(num_switch_states=%d, num_outputs=%d,\n" \
               "switch_trans_mat=%s,\nout_trans_mat=%s)" \
               %(self.num_switch_states,
                 self.num_outputs,
                 np.array_str(self.switch_trans_mat),
                 np.array_str(self.out_trans_mats))

    def __repr__(self):
        return self.__str__()

    def predict(self, num_steps):
        """
        Predict outputs and latent states.
        """
        if self.switch_state_trajectory is None:
            raise Exception, "Cannot predict; no hidden state trajectory."
        last_hidden_state = self.switch_state_trajectory[-1]
        # make a vector representation of last hidden state
        prev_state_probs = np.zeros(self.num_switch_states)
        prev_state_probs[last_switch_state] = 1.
        # predict transitions and subsequent outputs
        predictions = np.zeros(num_steps)
        # output probabilities
        predicted_probs = np.zeros((num_steps, self.num_outputs))
        for step in xrange(num_steps):
            # calculate probabilities of being in
            # states in next time step
            curr_state_probs = prev_state_probs.dot(self.trans_mat)
            predicted_probs[step, :] = curr_output_probs
            predictions[step] = np.random.multinomial(1, curr_output_probs).argmax()
            prev_state_probs = curr_state_probs
        return predictions

    def initialize(self):
        """
        initialize to random model.
        """
        # choose initial state probabilities
        self.init_switch_probs = np.random.dirichlet(self.init_switch_hyperparams)
        # choose switching transition matrix
        self.switch_trans_mat = init_trans_mat(self.switch_trans_mat_hyperparams)
        # choose observation matrix
        for s in xrange(self.num_switch_states):
            self.out_trans_mats[s, :] = \
              init_out_mat(self.out_trans_mat_hyperparams)
        self.initialized = True

    def add_data(self, data, init_switch_states=True):
        """
        Add data to model.
        """
        self.data_len = data.shape[0]
        self.outputs = data
        # if given data, initialize switch state trajectory
        if init_switch_states:
            self.switch_state_trajectory = np.zeros(self.data_len,
                                                    dtype=np.int32)
            # choose initial state
            self.switch_state_trajectory[0] = \
              np.random.multinomial(1, self.init_switch_probs).argmax()
            # choose switch state trajectories
            for n in xrange(1, self.data_len):
                prev_switch_state = self.switch_state_trajectory[n - 1]
                trans_mat = self.switch_trans_mat[prev_switch_state, :]
                self.switch_state_trajectory[n] = \
                  np.random.multinomial(1, trans_mat).argmax()

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
def sample_new_ssm(old_ssm, data):
    new_ssm = copy.deepcopy(old_ssm)
    # sample output matrix
    new_ssm.out_trans_mats = \
      sample_out_trans_mats(new_ssm.outputs,
                            new_ssm.switch_state_trajectory,
                            new_ssm.out_trans_mat_hyperparams,
                            new_ssm.num_switch_states)
    # sample switch states
    if DEBUG:
        print "old ssm: "
        print "-"*5
        print "old hidden state: "
        print new_ssm.switch_state_trajectory
        print "old trans mat: "
        print new_ssm.switch_trans_mat
        print "old out trans mat: "
        print new_ssm.out_trans_mats
    new_ssm.switch_state_trajectory = \
      sample_switch_states(new_ssm.switch_state_trajectory,
                           new_ssm.switch_trans_mat,
                           new_ssm.out_trans_mats,
                           new_ssm.outputs,
                           new_ssm.init_switch_probs,
                           new_ssm.init_switch_hyperparams)
    if DEBUG:
        print "NEW ssm: "
        print "*"*5
        print "new hidden state: "
        print new_ssm.switch_state_trajectory
        print "new trans mat: "
        print new_ssm.switch_trans_mat
        print "new out trans mat: "
        print new_ssm.out_trans_mats
        time.sleep(DEBUG_DELAY)
    # sample switch transition matrix
    new_ssm.switch_trans_mat = \
      sample_switch_trans_mat(new_ssm.switch_state_trajectory,
                              new_ssm.switch_trans_mat_hyperparams)
    # sample initial state probs
    new_ssm.init_switch_probs = \
      sample_init_state_prior(new_ssm.switch_state_trajectory[0],
                              new_ssm.init_switch_hyperparams)
    return new_ssm


##
## sample transition matrices
##
def sample_switch_trans_mat(switch_hyperparams, switch_states):
    """
    Sample P(switch transition matrix | switch_states, switch_hyperparams)

    By conjugacy, this is: Dir(counts(switch_states) + switch_hyperparams)
    """
    switch_counts = count_trans_mat(switch_states)
    switch_trans_mat = np.zeros((switch_hyperparams.shape[0],
                           switch_hyperparams.shape[1]))
    for n in xrange(switch_hyperparams.shape[0]):
        switch_trans_mat[n, :] = \
          random.dirichlet(switch_counts[n, :] + switch_hyperparams[n, :])
    return switch_trans_mat


def sample_init_state_prior(init_state, init_state_hyperparams):
    """
    Assume I is the Dirichlet parameter that determines
    value of the initial state S_1. Sample from its
    conditional distribution:

    P(I | S_1) \propto P(S_1 | I)P(I)
    """
    init_state_hyperparams = init_state_hyperparams.copy()
    init_state_hyperparams[init_state] += 1
    return np.random.dirichlet(init_state_hyperparams)


def count_out_mat(outputs, hidden_state_trajectory,
                  num_hidden_states,
                  num_possible_outputs):
    """
    Generate output counts matrix (num_hidden_states x num_possible_outputs).
    """
    count_mat = np.zeros((num_hidden_states, num_possible_outputs),
                         dtype=np.int32)
    seq_len = len(outputs)
    for n in xrange(seq_len):
        count_mat[hidden_state_trajectory[n], outputs[n]] += 1
    return count_mat


def sample_init_output(init_out_hyperparams):
    ##
    ## TODO: fill me in, sample P(Y_1 | \pi_Y, Y2, S1)
    ## here
    ##
    pass

##
## Sample switch states
##    
def sample_init_switch_state(init_switch_probs,
                             switch_state_trajectory,
                             outputs,
                             switch_trans_mat,
                             out_trans_mats):
    """
    Sample assignment of initial switch state.

    Args:
    -----
    - init_switch_probs: initial probabilities of switch states
    - switch_state_trajectory: switch state trajectory
    - outputs: observed outputs
    - switch_trans_mat: switch transition matrix
    - out_trans_mats: array of (n_switch_states x n_outputs x n_outputs)
      that gives the transition matrices between the outputs, for
      each possible switch state. Typically, a (2 x n_outputs x n_outputs)
      array.
    - next_state: next switch state

    log[P(S_1 | \pi_S, S_2, Y_1, Y_2, T_switch, T_out)] =
      # probability of initial switch state given prior
      log[P(S_1 | \pi_S)] +
      # probability of transitioning to next switch state
      # (if available) 
      log[P(S_2 | S_1, T_switch)] 
      # probability of transitioning from previous output
      # to next output given initial switch state
      log[P(Y_2 | Y_1, S_1, T_out)] 
    """
    possible_switch_states = np.arange(init_switch_probs.shape[0])
    log_scores = np.log(init_switch_probs)
    if switch_state_trajectory.shape[0] > 1:
        # probability of transitioning from first to second switch
        # state (only if there is a second switch state)
        log_scores += np.log(switch_trans_mat[possible_switch_states,
                                              switch_state_trajectory[1]])
        # probability of emitting transitioning from first
        # to second output under possible values of the initial
        # hidden switch state
        log_scores += np.log(out_trans_mats[possible_switch_states,
                                            outputs[0],
                                            outputs[1]])
    # sample value
    probs = np.exp(log_scores - scipy.misc.logsumexp(log_scores))
    sampled_init_switch_state = np.random.multinomial(1, probs).argmax()
    return sampled_init_switch_state

def sample_switch_states(switch_state_trajectory,
                         switch_trans_mat,
                         out_trans_mats,
                         outputs,
                         init_switch_probs,
                         init_switch_hyperparams):
    """
    Sample new configuration of switch states.

    Args:
    -----

    - switch_state_trajectory: trajectory of switch states
    - trans_mats: set of transition matrices (one for each switch state)
    - outputs: set of (dependent) outputs 

    Note that this modifies the 'switch_state_trajectory' array
    rather than making a copy.
    """
    num_switch_states = switch_trans_mat.shape[0]
    seq_len = switch_state_trajectory.shape[0]
    # sample initial switch state: P(S_1 | S_2, Y1, Y2, T, \pi_S)
    switch_state_trajectory[0] = \
      sample_init_switch_state(init_switch_probs,
                               switch_state_trajectory,
                               outputs,
                               switch_trans_mat,
                               out_trans_mats)
    possible_switch_states = np.arange(num_switch_states)
    for n in xrange(1, seq_len):
        prev_switch_state = switch_state_trajectory[n - 1]
        curr_output = outputs[n]
        # probability of transitioning from the previous
        # state to each of possible current states
        log_scores = np.log(switch_trans_mat[prev_switch_state,
                                             possible_switch_states])
        if (n + 1) < seq_len:
            # probability of transitioning to next switch state
            log_scores += \
              np.log(switch_trans_mat[possible_switch_states,
                                      switch_state_trajectory[n + 1]])
            # probability of emitting the next outputs given
            # the current output, for each possible value of
            # the current switch state
            next_output = outputs[n + 1]
            log_scores += \
              np.log(out_trans_mats[possible_switch_states,
                                    curr_output,
                                    next_output])
        # sample value
        probs = np.exp(log_scores - scipy.misc.logsumexp(log_scores))
        switch_state_trajectory[n] = np.random.multinomial(1, probs).argmax()
    return switch_state_trajectory

def sample_out_state(hidden_state, out_mat):
    out_state = np.random.multinomial(1, out_mat[hidden_state, :]).argmax()
    return out_state

def sample_switch_trans_mat(switch_state_trajectory,
                            switch_trans_mat_hyperparams):
    """
    Sample transition matrix.
    """
    # compute the matrix transition counts 
    switch_trans_mat_shape = (switch_trans_mat_hyperparams.shape[0],
                              switch_trans_mat_hyperparams.shape[0])
    counts_mat = \
      markov_utils.count_trans_mat(switch_state_trajectory,
                                   switch_trans_mat_shape)
    if DEBUG:
        print "counts mat: "
        print counts_mat
    # sample new transition matrix by iterating through
    # rows
    sampled_trans_mat = np.zeros(switch_trans_mat_shape)
    for n in xrange(sampled_trans_mat.shape[0]):
        # add counts from likelihood (counts matrix) and
        # from prior hyperparameters
        trans_row_params = counts_mat[n, :] + switch_trans_mat_hyperparams[n, :]
        sampled_trans_mat[n, :] = np.random.dirichlet(trans_row_params)
    return sampled_trans_mat

# Inference as adaptation in meta-changing environments
# Inference as adaptation in structured fluctuating environments

def sample_out_trans_mats(outputs,
                          switch_state_trajectory,
                          out_trans_mat_hyperparams,
                          num_switch_states):
    """
    Sample P(T_out | switch_states, outputs)
    """
    num_outputs = out_trans_mat_hyperparams.shape[1]
    sampled_out_trans_mats = np.zeros((num_switch_states,
                                       num_outputs,
                                       num_outputs))
    out_trans_mats_counts = np.zeros((num_switch_states,
                                      num_outputs,
                                      num_outputs))
    # compute transition matrix for each of the switch state values
    # in our trajectory separately
    seq_len = outputs.shape[0]
    for n in xrange(1, seq_len):
        # switch state in prior time step determines
        # transition to current output
        relevant_switch_state = switch_state_trajectory[n - 1]
        prev_output = outputs[n - 1]
        curr_output = outputs[n]
        # count transition
        out_trans_mats_counts[relevant_switch_state, prev_output, curr_output] += 1
    print "OUT MATS COUNTS: "
    print out_trans_mats_counts
    # for each of the possible switch state values,
    # sample output transition matrix. The output transition
    # matrices are sampled row by row (each row independent.)
    for switch_val in xrange(num_switch_states):
        for out_val in xrange(num_outputs):
            sampled_out_trans_mats[switch_val, out_val, :] = \
              np.random.dirichlet(out_trans_mats_counts[switch_val, out_val, :] + \
                                  out_trans_mat_hyperparams[out_val, :])
    return sampled_out_trans_mats

##
## scoring functions
##
def log_score_joint():
    """
    Score full joint model.
    """
    pass


def log_score_switch_state_trajectory(switch_state_trajectory,
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
