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
import particlefever.stat_utils as stat_utils
import particlefever.distributions as distributions

import scipy
import scipy.stats

DEBUG = False
DEBUG_DELAY = 0.05

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
        self.initialized = False
        self.num_hidden_states = num_hidden_states
        self.num_outputs = num_outputs
        self.hidden_state_trajectory = None
        self.out_state_trajectory = None
        self.trans_mat = np.zeros((self.num_hidden_states,
                                   self.num_hidden_states))
        self.out_mat = np.zeros((self.num_hidden_states,
                                 self.num_outputs))
        # hyperparameters for prior on initial state
        self.default_init_state_hyperparam = 1
        self.init_state_hyperparams = init_state_hyperparams
        if self.init_state_hyperparams is None:
            self.init_state_hyperparams = np.ones(self.num_hidden_states)
            self.init_state_hyperparams *= self.default_init_state_hyperparam
        # hyperparameters for prior on transition matrix
        self.trans_mat_hyperparams = trans_mat_hyperparams
        # hypeperameters for prior on output matrix
        self.out_mat_hyperparams = out_mat_hyperparams
        # default prior hyperparameters for transition and output
        # matrix
        self.default_trans_mat_hyperparam = 1.
        self.default_out_mat_hyperparam = 0.5
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
                 np.array_str(self.trans_mat, precision=3),
                 np.array_str(self.out_mat, precision=3))

    def __repr__(self):
        return self.__str__()

    def predict(self, num_steps):
        """
        Predict outputs and latent states.
        """
        if self.hidden_state_trajectory is None:
            raise Exception, "Cannot predict; no hidden state trajectory."
        last_hidden_state = self.hidden_state_trajectory[-1]
        # make a vector representation of last hidden state
        prev_state_probs = np.zeros(self.num_hidden_states)
        prev_state_probs[last_hidden_state] = 1.
        # predict transitions and subsequent outputs
        predictions = np.zeros(num_steps)
        # output probabilities
        predicted_probs = np.zeros((num_steps, self.num_outputs))
        for step in xrange(num_steps):
            # calculate probabilities of being in
            # states in next time step
            curr_state_probs = prev_state_probs.dot(self.trans_mat)
            # calculate output probabilities
            curr_output_probs = curr_state_probs.dot(self.out_mat)
            predicted_probs[step, :] = curr_output_probs
            predictions[step] = np.random.multinomial(1, curr_output_probs).argmax()
            prev_state_probs = curr_state_probs
        return predictions, predicted_probs

    def initialize(self):
        """
        Initialize to random model.
        """
        # choose initial state probabilities
        self.init_state_probs = np.random.dirichlet(self.init_state_hyperparams)
        # choose transition matrix
        self.trans_mat = init_trans_mat(self.trans_mat_hyperparams)
        # choose observation matrix
        self.out_mat = init_out_mat(self.out_mat_hyperparams)
        self.initialized = True

    def add_data(self, data, init_hidden_states=True):
        """
        Add data to model.
        """
        self.data_len = data.shape[0]
        self.outputs = data
        if init_hidden_states:
            self.hidden_state_trajectory = \
              init_hidden_state_trajectory(self.data_len,
                                           self.init_state_probs,
                                           self.trans_mat)


##
## counting functions
##
def count_out_mat(outputs, hidden_state_trajectory,
                  num_hidden_states,
                  num_possible_outputs):
    """
    Generate output counts matrix (num_hidden_states x num_possible_outputs).
    """
    count_mat = np.zeros((num_hidden_states, num_possible_outputs), dtype=np.int)
    seq_len = len(outputs)
    for n in xrange(seq_len):
        count_mat[hidden_state_trajectory[n], outputs[n]] += 1
    return count_mat

                  
##                  
## initialization functions
##
def init_hidden_state_trajectory(seq_len, init_state_probs, trans_mat,
                                 all_one_state=False):
    hidden_state_trajectory = np.zeros(seq_len, dtype=np.int)
    if all_one_state:
        return hidden_state_trajectory
    # choose initial state
    hidden_state_trajectory[0] = \
      np.random.multinomial(1, init_state_probs).argmax()
    # choose hidden state trajectories
    for n in xrange(1, seq_len):
        prev_hidden_state = hidden_state_trajectory[n - 1]
        hidden_state_trajectory[n] = \
          np.random.multinomial(1, trans_mat[prev_hidden_state, :]).argmax()
    return hidden_state_trajectory

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
## conditional distributions (for Gibbs sampling)
##
def cond_init_state_prior(init_state, init_state_hyperparams):
    """
    Assume I is the Dirichlet parameter that determines
    value of the initial state S_1.
    P(I | S_1) \propto P(S_1 | I)P(I)
    """
    init_state_hyperparams = init_state_hyperparams.copy()
    init_state_hyperparams[init_state] += 1
    return distributions.Dirichlet(init_state_hyperparams)

def cond_init_state(init_state_probs,
                    hidden_state_trajectory,
                    outputs,
                    trans_mat,
                    out_mat):
    """
    Sample assignment of initial state prior given initial state
    hyperparameters.

    P(S_0 | I, Y_0, O_mat) = P(S_0 | I)P(Y_0 | S_0)P(S_1 | S_0, T_mat)P(T_mat)
    """
    possible_hidden_states = np.arange(len(init_state_probs))
    log_scores = \
      np.log(init_state_probs) + \
      np.log(out_mat[possible_hidden_states, outputs[0]])
    if hidden_state_trajectory.shape[0] > 1:
        # include next state if present
        log_scores += np.log(trans_mat[possible_hidden_states,
                                       hidden_state_trajectory[1]])
    return distributions.LogMultinomial(log_scores)

##    
## sampling functions
##
def sample_new_hmm(old_hmm, data):
    new_hmm = copy.deepcopy(old_hmm)
    # sample output matrix
    new_hmm.out_mat = cond_out_mat(new_hmm.outputs,
                                   new_hmm.hidden_state_trajectory,
                                   new_hmm.out_mat_hyperparams,
                                   new_hmm.num_hidden_states).sample()
    # sample hidden states
    if DEBUG:
        print "OLD HMM: "
        print "-"*5
        print "old hidden state: "
        print new_hmm.hidden_state_trajectory
        print "old trans mat: "
        print new_hmm.trans_mat
        print "old out mat: "
        print new_hmm.out_mat
    new_hmm.hidden_state_trajectory = \
      sample_hidden_states(new_hmm.hidden_state_trajectory,
                           new_hmm.trans_mat,
                           new_hmm.out_mat,
                           new_hmm.outputs,
                           new_hmm.init_state_probs,
                           new_hmm.init_state_hyperparams)
    if DEBUG:
        print "NEW HMM: "
        print "*"*5
        print "new hidden state: "
        print new_hmm.hidden_state_trajectory
        print "new trans mat: "
        print new_hmm.trans_mat
        print "new out mat: "
        print new_hmm.out_mat
        time.sleep(DEBUG_DELAY)
    # sample transition matrix
    new_hmm.trans_mat = cond_trans_mat(new_hmm.hidden_state_trajectory,
                                       new_hmm.trans_mat_hyperparams).sample()
    # sample initial state probs
    new_hmm.init_state_probs = \
      cond_init_state_prior(new_hmm.hidden_state_trajectory[0],
                            new_hmm.init_state_hyperparams).sample()
    return new_hmm

def cond_hidden_state(state_ind,
                      hidden_state_trajectory,
                      trans_mat,
                      out_mat,
                      outputs,
                      init_state_probs,
                      init_state_hyperparams):
    """
    Conditional distribution of a hidden state.
    """
    num_hidden_states = trans_mat.shape[0]
    seq_len = hidden_state_trajectory.shape[0]
    if state_ind == 0:
        # it's the initial state
        init_cond_dist = cond_init_state(init_state_probs,
                                         hidden_state_trajectory,
                                         outputs,
                                         trans_mat,
                                         out_mat)
        return init_cond_dist
    # otherwise, it's a middle (or end) state
    possible_hidden_states = np.arange(len(init_state_probs))
    prev_hidden_state = hidden_state_trajectory[state_ind - 1]
    log_scores = \
      np.log(trans_mat[prev_hidden_state, possible_hidden_states]) + \
      np.log(out_mat[possible_hidden_states, outputs[state_ind]])
    if (state_ind + 1) < seq_len:
        # P(S_1 | S_0): probability of transitioning to next state, if
        # available
        log_scores += np.log(trans_mat[possible_hidden_states,
                                       hidden_state_trajectory[state_ind + 1]])
    return distributions.LogMultinomial(log_scores)

def sample_hidden_states(hidden_state_trajectory,
                         trans_mat,
                         out_mat,
                         outputs,
                         init_state_probs,
                         init_state_hyperparams):
    """
    Sample new configuration of hidden states.
    """
    num_hidden_states = trans_mat.shape[0]
    seq_len = hidden_state_trajectory.shape[0]
    for state_ind in xrange(seq_len):
        state_cond_dist = cond_hidden_state(state_ind,
                                            hidden_state_trajectory,
                                            trans_mat,
                                            out_mat,
                                            outputs,
                                            init_state_probs,
                                            init_state_hyperparams)
        hidden_state_trajectory[state_ind] = state_cond_dist.sample()
    return hidden_state_trajectory

def sample_out_state(hidden_state, out_mat):
    """
    Sampling output state. Used for initialization of models; not part
    of Gibbs sampling moves.
    """
    out_state = np.random.multinomial(1, out_mat[hidden_state, :]).argmax()
    return out_state

def cond_trans_mat(hidden_state_trajectory,
                   trans_mat_hyperparams):
    """
    Conditional distribution of transition matrix given
    hidden state trajectory and transition matrix hyperparameters.
    """
    # compute the matrix transition counts 
    trans_mat_shape = (trans_mat_hyperparams.shape[0],
                       trans_mat_hyperparams.shape[0])
    counts_mat = \
      markov_utils.count_trans_mat(hidden_state_trajectory,
                                   trans_mat_shape)
    # posterior matrix with counts and hyperparameters (pseudocounts)
    updated_mat = counts_mat + trans_mat_hyperparams
    return distributions.DirichletMatrix(updated_mat)

def cond_out_mat(outputs,
                 hidden_state_trajectory,
                 out_mat_hyperparams,
                 num_hidden_states):
    """
    Conditional distribution of output matrix given
    hidden state trajectory and output matrix hyperparameters.
    """
    num_outputs = out_mat_hyperparams.shape[1]
    # number of occurences of each output state
    out_counts = count_out_mat(outputs, hidden_state_trajectory,
                               num_hidden_states, num_outputs)
    updated_mat = out_counts + out_mat_hyperparams
    return distributions.DirichletMatrix(updated_mat)

##
## particle filter related classes and conditional distributions
##
class ParticlePrior:
    """
    Prior for discrete Bayes HMM.
    """
    def __init__(self, num_hidden_states, num_outputs):
        self.hmm = DiscreteBayesHMM(num_hidden_states, num_outputs)

    def initialize(self, num_particles):
        """
        Initialize set of particles.
        """
        particles = []
        for n in xrange(num_particles):
            init_state_probs = np.random.dirichlet(self.hmm.init_state_hyperparams)
            # initialize hidden state
            hidden_state = np.random.multinomial(1, init_state_probs).argmax()
            # initialize particle with hidden state
            particle = Particle(hidden_state,
                                self.hmm.trans_mat_hyperparams,
                                self.hmm.out_mat_hyperparams)
            particles.append(particle)
        return particles

class Particle:
    """
    Particle for discrete Bayes HMM.
    """
    def __init__(self, hidden_state, hidden_trans_counts, output_counts):
        # hidden state: the variable to be estimated
        self.hidden_state = hidden_state
        # counters for hidden state transitions and outputs
        self.hidden_trans_counts = hidden_trans_counts
        self.output_counts = output_counts

def pf_prior():
    """
    Make Bayes HMM prior for particle filter.
    """
    return ParticlePrior()

def pf_trans_sample(prev_particle, prior):
    """
    Sample transition to new particle given previous particle.
    """
    # sample transition to new particle using posterior predictive
    # dirichlet distribution
    new_particle = copy.deepcopy(prev_particle)
    # get the relevant counts (i.e. counts conditioned on the previous state)
    prev_counts = \
      prev_particle.hidden_trans_counts[prev_particle.hidden_state, :]
    # prior counts for the each of the possible hidden states
    # given the previous hidden state
    hidden_state_prior_counts = \
      prior.hmm.trans_mat_hyperparams[prev_particle.hidden_state, :]
    log_scores = np.log(prev_counts + hidden_state_prior_counts)
    if (log_scores == -np.inf).any():
        print "WARNING: transition probability to new particle contained -inf"
    # sample new hidden state
    next_state_dist = distributions.DirMultinomial(prev_counts,
                                                   hidden_state_prior_counts)
    new_particle.hidden_state = next_state_dist.sample_posterior_pred()
    # update the particle's state transition count
    new_particle.hidden_trans_counts[prev_particle.hidden_state,
                                     new_particle.hidden_state] += 1
    return new_particle
    
def pf_observe(data_point, particle, prior):
    """
    Evaluate P(output | particle) (i.e. P(obs | S)).
    Returns:
      - weight for this data point given its particle.
    """
    out_prior_counts = prior.hmm.out_mat_hyperparams[particle.hidden_state, :]
    # weigh the particle by its probability
    out_dist = \
      distributions.DirMultinomial(particle.output_counts[particle.hidden_state, :],
                                   out_prior_counts)
    weight = out_dist.posterior_pred_pmf(data_point)
    # update the particle counts
    particle.output_counts[particle.hidden_state, data_point] += 1
    return weight

##
## scoring functions
##
def log_score_joint(hmm_obj):
    """
    Score full joint model.
    """
    init_state = hmm_obj.hidden_state_trajectory[0]
    init_probs = hmm_obj.init_probs
    hidden_state_trajectory = hmm_obj.hidden_state_trajectory
    outputs = hmm_obj.outputs
    trans_mat = hmm_obj.trans_mat 
    out_mat = hmm_obj.out_mat
    out_mat_hyperparams = hmm_obj.out_mat_hyperparams
    trans_mat_hyperparams = hmm_obj.trans_mat_hyperparams
    # score hidden states
    log_hidden_trajectory = \
      log_score_hidden_state_trajectory(hidden_state_trajectory,
                                        outputs,
                                        trans_mat,
                                        out_mat,
                                        init_probs)
    # score transition matrix
    log_trans_mat = \
      log_score_trans_mat(trans_mat, trans_mat_hyperparams)
    # score outputs
    log_outputs = \
      np.sum(np.log(out_mat[hidden_state_trajectory, outputs]))
    # score output matrix
    log_out_mat = \
      log_score_out_mat(out_mat, out_mat_hyperparams)
    if DEBUG:
        print "log total: "
        print "*" * 5
        print "log hidden traj: ", log_hidden_trajectory
        print "log outputs: ", log_outputs
        print "log_trans mat: ", log_trans_mat
        print "log_out mat: ", log_out_mat
    log_total = \
      log_hidden_trajectory + \
      log_outputs + \
      log_trans_mat + \
      log_out_mat 
    return log_total

def log_score_outputs(hidden_state_trajectory,
                      outputs,
                      out_mat):
    return np.sum(np.log(out_mat[hidden_state_trajectory, outputs]))

def log_score_out_mat(out_mat, out_mat_hyperparams):
    log_score = 0.
    for n in xrange(out_mat.shape[0]):
        log_score += scipy.stats.dirichlet.logpdf(out_mat[n, :],
                                                  out_mat_hyperparams[n, :])
    return log_score

def log_score_trans_mat(trans_mat, trans_mat_hyperparams):
    log_score = 0.
    for n in xrange(trans_mat.shape[0]):
        log_score += scipy.stats.dirichlet.logpdf(trans_mat[n, :],
                                                  trans_mat_hyperparams[n, :])
    return log_score

def log_score_hidden_state_trajectory(hidden_trajectory,
                                      outputs,
                                      trans_mat,
                                      out_mat,
                                      init_probs):
    """
    Score hidden state assignment of a single node, given observations,
    transition matrix, output matrix, and initial state probabilities.
    Return a vector of log scores:

    log[P(hidden_trajectory | outputs, trans_mat, obs_mat, init_probs])
    """
    num_obs = outputs.shape[0]
    log_scores = np.zeros(num_obs, dtype=np.float64)
    # handle special case of single observation
    # the score is the probability of being
    # in this initial state and emitting the single observation
    init_hidden_state = hidden_trajectory[0]
    # initial observation
    init_out = outputs[0]
    if num_obs == 1:
        # prob. of being in this initial state
        log_scores[0] = \
          np.log(init_probs[init_hidden_state])
        return np.sum(log_scores)
    # score initial state: it depends on its output and
    # the next hidden state.
    # prob. of being in initial hidden state times 
    # prob. of observing output given initial state
    log_scores[0] = \
      np.log(init_probs[init_hidden_state])
    ### vectorized version
    log_scores[1:] = \
      np.log(trans_mat[hidden_trajectory[0:-1],
                       hidden_trajectory[1:]])
    return np.sum(log_scores)

if __name__ == "__main__":
    trans_mat = np.matrix([[0.9, 0.1],
                           [0.1, 0.9]])
    out_mat = np.matrix([0.5, 0.5])
    #DiscreteBayesHMM()
