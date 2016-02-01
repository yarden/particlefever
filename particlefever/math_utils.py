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


def log_multinomial_coeff(counts):
    return log_factorial(sum(counts)) - \
           sum(log_factorial(c) for c in counts)

def log_factorial(num):
    if not round(num) == num and num > 0:
        raise ValueError("Can only compute the factorial of positive ints")
    return sum(math.log(n) for n in xrange(1,num+1))

def multinomial_logpmf(counts, params):
    if not (len(counts) == len(params)):
        raise ValueError("Dimensionality of count vector is incorrect")
    prob = 0.
    #for i, c in enumerate(counts):
    for i in xrange(len(counts)):
        prob += counts[i] * math.log(params[i])
    return prob + log_multinomial_coeff(counts)

def multinomial_pmf(counts, params):
    if not(len(counts) == len(params)):
        raise ValueError("Dimensionality of count vector is incorrect")
    prob = 1.
    #for i,c in enumerate(counts):
    for i in xrange(len(counts)):
        prob *= params[i]**counts[i]
    return prob * math.exp(log_multinomial_coeff(counts))

