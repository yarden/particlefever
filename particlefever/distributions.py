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

class LogMultinomial(Distribution):
    """
    Multinomial that works with log probabilities.
    """
    def __init__(self, logp):
        self.logp = logp
        self.p = np.exp(self.logp - stat_utils.logsumexp(self.logp))

    def pmf(self):
        pass

    def logpmf(self):
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
            log_score += scipy.stats.dirichlet.logpdf(self.alpha_mat[n, :])
        return log_score

    def sample(self):
        ##
        ## TODO: can vectorize this sampling
        ##
        new_T = np.zeros(self.alpha_mat.shape)
        for n in xrange(self.alpha_mat.shape[0]):
            new_T[n, :] = np.random.dirichlet(self.alpha_mat[n, :])
        return new_T
