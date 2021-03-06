##
## Distributions.
##
import numpy as np
import scipy
import scipy.stats

import particlefever
import particlefever.stat_utils as stat_utils

class Distribution:
    def __init__(self):
        pass

    def sample(self):
        pass
    
    def rvs(self):
        """
        Same as self.sample(): to be consistent with scipy's convention.
        """
        return self.sample()


class DirMultinomial(Distribution):
    """
    Dirichlet-Multinomial distribution.
    """
    def __init__(self, prev_counts, alpha):
        self.prev_counts = prev_counts
        self.alpha = alpha
        # posterior predictive distribution
        # note: do not use logsumexp here for denominator, since
        # the counts were never logged! this is just 'logsum'
        self.log_posterior_pred = \
          np.log(self.prev_counts + self.alpha) - \
          np.log(np.sum(self.prev_counts + self.alpha))
        self.posterior_pred = np.exp(self.log_posterior_pred)
        
    def log_posterior_pred_pmf(self, ind):
        """
        Log score of posterior predictive distribution for Dirichlet-Multinomial
        distribution. Scores a state value index 'ind'.

        If X_n represents the value of the next sample and X is the set of
        previous observations, then this is:

        P(X_n+1 = ind | X, alpha) = \int[P(X_n+1 = ind | p)P(p | X, alpha)dp]
        """
        return self.log_posterior_pred[ind]

    def posterior_pred_pmf(self, ind):
        return self.posterior_pred[ind]

    def sample_posterior_pred(self):
        return np.random.multinomial(1, self.posterior_pred).argmax()


class LogMultinomial(Distribution):
    """
    Multinomial that works with log probabilities.
    """
    def __init__(self, logp):
        self.logp = logp
        self.p = np.exp(self.logp - stat_utils.logsumexp(self.logp))

    def pmf(self, ind):
        pass

    def logpmf(self, ind):
        pass

    def sample(self):
        return np.random.multinomial(1, self.p).argmax()

    def __str__(self):
        return "LogMultinomial(logp=%s)" %(np.array_str(self.logp, precision=4))


class Dirichlet(Distribution):
    """
    Dirichlet distribution.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def pdf(self, x):
        return scipy.stats.dirichlet.pdf(x, self.alpha)

    def logpdf(self, x):
        return scipy.stats.dirichlet.logpdf(x, self.alpha)

    def sample(self):
        return np.random.dirichlet(self.alpha)

    def __str__(self):
        return "Dirichlet(alpha=%s)" %(np.array_str(self.alpha, precision=4))


class DirichletMatrix(Distribution):
    """
    Dirichlet matrix distribution. Assumes each row
    follows an independent Dirichlet distribution.

    Used as prior on transition matrices.
    """
    def __init__(self, alpha_mat):
        self.alpha_mat = alpha_mat

    def pdf(self, T):
        pass

    def logpdf(self, T):
        ###
        ### TODO: can vectorize this scoring
        ##
        log_score = 0.
        for n in xrange(self.alpha_mat.shape[0]):
            # normalize T (by row) in case it's a matrix of counts
            log_score += scipy.stats.dirichlet.logpdf(T[n, :] / T[n, :].sum(), self.alpha_mat[n, :])
        return log_score

    def sample(self):
        ##
        ## TODO: can vectorize this sampling
        ##
        new_T = np.zeros(self.alpha_mat.shape)
        for n in xrange(self.alpha_mat.shape[0]):
            new_T[n, :] = np.random.dirichlet(self.alpha_mat[n, :])
        return new_T

    def __str__(self):
        return "DirichletMatrix(alpha_mat=%s)" %(np.array_str(self.alpha_mat,
                                                              precision=4))
