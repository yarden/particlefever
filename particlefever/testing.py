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
import particlefever.sampler as sampler

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


class TestDiscreteBayesHMM(unittest.TestCase):
    """
    Test discrete Bayesian HMM scoring functions.
    """
    def setUp(self):
        trans_mat_hyperparams = np.ones((2, 2))
        trans_mat_hyperparams *= 0.1
        # put peaky prior on outputs
        out_mat_hyperparams = np.ones((2, 2))
        out_mat_hyperparams *= 0.1
        self.simple_hmm = \
          bayes_hmm.DiscreteBayesHMM(2, 2,
                                     trans_mat_hyperparams=trans_mat_hyperparams,
                                     out_mat_hyperparams=out_mat_hyperparams)
        
    def test_init_hmm(self):
        """
        Test initialization of an HMM.
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
        # initialize 3-state 2-output HMM
        num_hidden_states = 3
        num_outputs = 2
        default_hmm = bayes_hmm.DiscreteBayesHMM(num_hidden_states,
                                                 num_outputs)
        print "3-state 2-output HMM: "
        print default_hmm
        default_hmm.initialize()
        print default_hmm

    def test_out_mat_sampling(self):
        outputs = np.array([0, 1, 0, 1, 0, 1])
        hidden_state_trajectory = np.array([1, 1, 1, 1, 1, 1])
        out_mat_hyperparams = np.array([[0.3, 0.3],
                                        [0.3, 0.3]])
        num_hidden_states = 2
        out_mat = bayes_hmm.sample_out_mat(outputs,
                                           hidden_state_trajectory,
                                           out_mat_hyperparams,
                                           num_hidden_states)
        print "out_mat: "
        print out_mat
        

    def _test_gibbs_inference(self):
        """
        Test Gibbs sampling in HMM.
        """
        data = np.array([0, 1]*20 + [1, 1]*20)
        gibbs_obj = sampler.DiscreteBayesHMMGibbs(self.simple_hmm)
        gibbs_obj.sample(data)
        
    def get_mean_preds(self, samples, num_preds=20):
        all_preds = []
        for curr_hmm in samples:
            curr_hmm = samples[-1]
            preds = curr_hmm.predict(num_preds)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        mean_preds = all_preds.mean(axis=0)
        return mean_preds
        
    # def test_hmm_prediction(self):
    #     # clamp seed to debug
    #     #np.random.seed(3)
    #     # data biased toward state 1
    #     data = np.array([1, 1]*5)
    #     print "data: ", data
    #     gibbs_obj = sampler.DiscreteBayesHMMGibbs(self.simple_hmm)
    #     gibbs_obj.sample(data)
    #     mean_preds = self.get_mean_preds(gibbs_obj.samples)
    #     print "predicting using sampled HMM"
    #     print "mean preds: ", mean_preds
    #     assert (mean_preds > 0.7).all(), \
    #       "Expected all predictions to be output 1 with prob. > 0.7"
    #     # compare this to an HMM with biased data toward state 0
    #     data = np.array([0, 0]*5)
    #     print "data: ", data
    #     gibbs_obj = sampler.DiscreteBayesHMMGibbs(self.simple_hmm)
    #     gibbs_obj.sample(data)
    #     mean_preds = self.get_mean_preds(gibbs_obj.samples)
    #     print "predicting using sampled HMM"
    #     print "mean preds: ", mean_preds
    #     assert (mean_preds < 0.3).all(), \
    #       "Expected all predictions to be output 1 with prob. < 0.3"

    def test_hmm_prediction_periodic(self):
        print "\ntesting periodic predictions: "
        # now test it with a periodic data set
        data = np.array([0, 1]*20)
        print "data: ", data
        gibbs_obj = sampler.DiscreteBayesHMMGibbs(self.simple_hmm)
        gibbs_obj.sample(data)
        mean_preds = self.get_mean_preds(gibbs_obj.samples)
        print "predicting using sampled HMM"
        print "mean preds: ", mean_preds
        print " --- "
        raise Exception, "test"
        
        
        
        

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

