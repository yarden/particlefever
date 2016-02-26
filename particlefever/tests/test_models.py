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

import copy

import particlefever
import particlefever.math_utils as math_utils
import particlefever.bayes_hmm as bayes_hmm
import particlefever.sampler as sampler
import particlefever.switch_ssm as switch_ssm
import particlefever.particle_filter as particle_filter

class TestDiscreteSwitchSSM(unittest.TestCase):
    def setUp(self):
        self.simple_ssm = switch_ssm.DiscreteSwitchSSM(2, 2)
        
    def _test_sample_mats(self):
        print "testing sampling of matrices"
        data = np.array([0, 1] * 50)# + [1, 1] * 5)
        self.simple_ssm.initialize()
        self.simple_ssm.add_data(data)
        switch_ssm.sample_new_ssm(self.simple_ssm, data)

    def _test_gibbs(self):
        print "testing Gibbs sampling"
        data = np.array([0, 1] * 50)# + [1, 1] * 10)
        ssm = copy.deepcopy(self.simple_ssm)
        gibbs_obj = sampler.DiscreteSwitchSSMGibbs(ssm)
        gibbs_obj.sample(data, num_iters=2000, burn_in=100)
        num_preds = 10
        pred_probs = switch_ssm.get_predictions(gibbs_obj.samples, num_preds)
        print "predicting next %d obs: " %(num_preds)
        print pred_probs

    def test_filter_fit(self):
        print "testing filter fitting"
        data = np.array([0, 1] * 10)
        ssm = copy.deepcopy(self.simple_ssm)
        gibbs_obj = sampler.DiscreteSwitchSSMGibbs(ssm)
        gibbs_obj.filter_fit(ssm, data, switch_ssm.get_predictions,
                             num_iters=2000, burn_in=100)
        print gibbs_obj.get_prediction_probs(num_outputs=ssm.num_outputs)
        

class TestDiscreteBayesHMM(unittest.TestCase):
    """
    Test discrete Bayesian HMM scoring functions.
    """
    def setUp(self):
        trans_mat_hyperparams = np.ones((2, 2))
        trans_mat_hyperparams *= 1.
        #trans_mat_hyperparams *= 1.
        # put peaky prior on outputs
        out_mat_hyperparams = np.ones((2, 2))
        #out_mat_hyperparams *= 3.739
        out_mat_hyperparams *= 0.5
        self.simple_hmm = \
          bayes_hmm.DiscreteBayesHMM(2, 2,
                                     trans_mat_hyperparams=trans_mat_hyperparams,
                                     out_mat_hyperparams=out_mat_hyperparams)
        
    def _test_init_hmm(self):
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


    def test_log_joint_scores(self):
        """
        Test scoring of joint distribution.
        """
        num_pairs = 10
        outputs = np.array([0, 1] * num_pairs)
        trans_mat_hyperparams = np.ones((2, 2))
        trans_mat_hyperparams *= 1.
        out_mat_hyperparams = np.ones((2, 2))
        out_mat_hyperparams *= 1.
        
        hmm1 = copy.deepcopy(self.simple_hmm)
        hmm1.trans_mat_hyperparams = trans_mat_hyperparams
        hmm1.out_mat_hyperparams = out_mat_hyperparams
        hmm1.hidden_state_trajectory = np.array([0, 1] * num_pairs)
        # deterministic constant switching
        # with peaked output for each state
        zero = np.power(10., -5)
        one = 1. - zero
        hmm1.init_probs = np.array([0.5, 0.5])
        hmm1.trans_mat = np.array([[zero, one],
                                   [one, zero]])
        hmm1.out_mat = np.array([[one, zero],
                                 [zero, one]])
        hmm1.outputs = outputs
        # no hidden state switching, with 0.5
        # probability of output
        hmm2 = copy.deepcopy(hmm1)
        hmm2.hidden_state_trajectory = np.array([0, 0] * num_pairs)
        hmm2.init_probs = hmm1.init_probs
        hmm2.trans_mat = np.array([[one, zero],
                                   [zero, one]])
        hmm2.out_mat = np.array([[0.5, 0.5],
                                 [one, zero]])
        hmm2.outputs = outputs
        # score the two models
        log_score1 = bayes_hmm.log_score_joint(hmm1)
        log_score2 = bayes_hmm.log_score_joint(hmm2)
        print "testing log joint scores: "
        print "Periodic HMM: "
        print hmm1
        print "log score: %.4f" %(log_score1)
        print "==" * 5
        print "Equiprobable HMM: "
        print hmm2
        print "log score: %.4f" %(log_score2)
        print "Ratio of log score 1 to log score 2: "
        print " - diff: %.4f" %(log_score1 - log_score2)
        print " - ratio: %.6f" %(np.exp(log_score1) / np.exp(log_score2))
        assert (log_score1 > log_score2), \
          "Periodic hypothesis should be favored."

    def get_mean_preds(self, samples, num_preds=20, num_outputs=2):
        num_samples = len(samples)
        all_preds = np.zeros((num_samples, num_preds, num_outputs))
        n = 0
        for n in xrange(num_samples):
            curr_hmm = samples[-1]
            preds, pred_probs = curr_hmm.predict(num_preds)
            all_preds[n, :] = pred_probs
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

    def _test_hmm_prediction_periodic(self):
        print "\n"
        print "testing periodic predictions: "
        # now test it with a periodic data set
        data = np.array([0, 1] * 20)
        print "data: ", data
        hmm_obj = copy.deepcopy(self.simple_hmm)
        # initialize it with the right configuration
        #hmm_obj.hidden_state_trajectory = copy.copy(data)
        hmm_obj.hidden_state_trajectory = np.array([0, 0] * 20)
        hmm_obj.init_state_probs = np.array([0.5, 0.5])
        hmm_obj.trans_mat = np.array([[0.1, 0.9],
                                      [0.9, 0.1]])
        hmm_obj.out_mat = np.array([[0.9, 0.1],
                                    [0.1, 0.9]])
        gibbs_obj = sampler.DiscreteBayesHMMGibbs(hmm_obj)
        gibbs_obj.sample(data, init_hidden_states=False)
        mean_preds = self.get_mean_preds(gibbs_obj.samples)
        print "predicting using sampled HMM"
        print "mean preds: ", mean_preds

    def test_hmm_particle_filter(self):
        """
        Test particle filter.
        """
        num_hidden_states = 2
        num_outputs = 2
        num_particles = 50
        print "Setting seed!"
        np.random.seed(50)
        hmm_pf = particle_filter.DiscreteBayesHMM_PF(num_hidden_states,
                                                     num_outputs,
                                                     num_particles=num_particles)
        hmm_pf.initialize()
        data = np.array([0, 1] * 50)
        #data = np.array([0, 0] * 50)
        print "HMM PF: "
        print hmm_pf
        hmm_pf.process_data(data)
        print "predicting output..."
        hmm_pf.predict_output()

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

