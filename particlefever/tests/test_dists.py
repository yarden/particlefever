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
import particlefever.distributions as distributions

class TestDistributions(unittest.TestCase):
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

    def test_dir_multinomial(self):
        """
        Test Dirichlet-Multinomial distribution.
        """
        p = np.array([0.2, 0.1, 0.7])
        N = 10000
        # sample random counts
        counts = np.random.multinomial(N, p)
        state_ind = 1
        # get empirical estimate of posterior predictive distribution
        # of the state
        num_iters = 100000
        num_occurs_ind = 0
        num_occurs = np.zeros(len(p))
        for n in xrange(num_iters):
            num_occurs += np.random.multinomial(1, p)
        # empirical estimate of posterior predictive
        emp_posterior_pred = num_occurs / float(num_iters)
        for state_ind in xrange(len(p)):
            print "--"
            print "Empirical estimate of posterior predictive distribution"
            print "P(X = %d | %s): %.4f" %(state_ind,
                                           np.array_str(p, precision=4),
                                           emp_posterior_pred[state_ind])
            print "Analytic calculation of posterior predictive distribution"
            alpha = np.ones(len(p))
            dir_mult = distributions.DirMultinomial(counts, alpha)
            logp = dir_mult.log_posterior_pred_pmf(state_ind)
            print "P(X = %d | %s): %.4f" %(state_ind,
                                           np.array_str(counts, precision=4),
                                           np.exp(logp))
            # since we're using non-zero alpha values, it shouldn't be
            # an exact match
            abs_error = 0.02
            assert (np.allclose(emp_posterior_pred[state_ind],
                                np.exp(logp), atol=abs_error)), \
                   "Empirical and analytic calculation don't match."


if __name__ == "__main__":
    unittest.main()
