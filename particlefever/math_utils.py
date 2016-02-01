## 
## Math utilities
##
import math
import numpy as np

import particlefever

def sample_multinomial_logprobs(log_probs):
    """
    Sample multinomial from log probabilities.

    Based on:
    http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point
    """
    ####
    #### TODO: check that this is correct
    ####
    p_max = np.max(log_probs)
    log_norm_factor = p_max + np.log(np.sum(np.exp(log_probs - p_max)))
    norm_probs = np.exp(log_probs - log_norm_factor)
    results = np.random.multinomial(1, norm_probs)
    ind = results.nonzero()[0]
    if len(ind) != 0:
        return ind[0]
    return np.nan


class Multinomial(object):
    def __init__(self, params):
        self._params = params

    def pmf(self, counts):
        if not(len(counts)==len(self._params)):
            raise ValueError("Dimensionality of count vector is incorrect")
        prob = 1.
        for i,c in enumerate(counts):
            prob *= self._params[i]**counts[i]
        return prob * math.exp(self._log_multinomial_coeff(counts))

    def log_pmf(self,counts):
        if not(len(counts)==len(self._params)):
            raise ValueError("Dimensionality of count vector is incorrect")
        prob = 0.
        for i,c in enumerate(counts):
            prob += counts[i]*math.log(self._params[i])
        return prob + self._log_multinomial_coeff(counts)

    def _log_multinomial_coeff(self, counts):
        return self._log_factorial(sum(counts)) - \
               sum(self._log_factorial(c) for c in counts)

    def _log_factorial(self, num):
        if not round(num)==num and num > 0:
            raise ValueError("Can only compute the factorial of positive ints")
        return sum(math.log(n) for n in range(1,num+1))



