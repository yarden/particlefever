##
## Testing
##
import os
import sys
import time
import unittest
import numpy as np
import scipy
import scipy.stats 

import particlefever
import particlefever.math_utils as math_utils
import particlefever.bayes_hmm as bayes_hmm

class TestGeneralScoring(unittest.TestCase):
    """
    Test general scoring functions.
    """
    def test_multinomial_score(self):
        """
        Test multinomial scoring function
        """
        # test multinomial scoring function
        n = 10
        val_multi = math_utils.multinomial_pmf([5, 5], [0.5, 0.5])
        val_binom = scipy.stats.binom.pmf(5, 10, 0.5)
        assert (np.isclose(val_multi, val_binom)), \
          "multinomial and binomial pmfs give different answers."
        # test scoring in log space
        val_multi = math_utils.multinomial_logpmf([5, 5], [0.5, 0.5])
        val_binom = scipy.stats.binom.logpmf(5, 10, 0.5)
        assert (np.isclose(val_multi, val_binom)), \
          "multinomial and binomial logpmfs give different answers."


class TestDiscreteBayesHMMScoring(unittest.TestCase):
    """
    Test discrete Bayesian HMM scoring functions.
    """
    def test_init_hmm(self):
        """
        Test initialization of an HMM
        """
        # initialize 2-state 2-output HMM
        num_hidden_states = 2
        num_outputs = 2
        default_hmm = bayes_hmm.DiscreteBayesHMM(num_hidden_states,
                                                 num_outputs)
        print "2-state 2-output HMM: "
        print default_hmm
        print "initializing: "
        default_hmm.initialize()
        print default_hmm
        # initialize 2-state 3-output HMM
        num_hidden_states = 2
        num_outputs = 3
        default_hmm = bayes_hmm.DiscreteBayesHMM(num_hidden_states,
                                                 num_outputs)
        print "2-state 3-output HMM: "
        print default_hmm
        default_hmm.initialize()
        print default_hmm
        
    def test_score_hidden_state_trajectory(self):
        """
        Test scoring of hidden state trajectory.
        """
        trans_mat = np.matrix([[0.9, 0.1],
                               [0.2, 0.8]])
        out_mat = np.matrix([[0.5, 0.5],
                             [0.9, 0.1]])
        init_probs = np.array([0.7, 0.3])
        # score singleton observations
        observations = np.array([0])
        log_score1 = \
          bayes_hmm.log_score_hidden_state_trajectory(np.array([0]),
                                                      observations,
                                                      trans_mat,
                                                      out_mat,
                                                      init_probs)
        log_score2 = \
          bayes_hmm.log_score_hidden_state_trajectory(np.array([1]),
                                                      observations,
                                                      trans_mat,
                                                      out_mat,
                                                      init_probs)
        assert (log_score1 > log_score2), "Initial hidden state 0 more favorable."
        # score larger observations
        observations1 = np.array([0, 1, 0, 1])
        
          
        

if __name__ == "__main__":
    unittest.main()

