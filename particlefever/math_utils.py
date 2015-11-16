## 
## Math utilities
##
import numpy as np

import particlefever

def sample_multinomial_logprobs(log_probs):
    """
    Sample multinomial from log probabilities.

    Based on:
    http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point
    """
    p_max = np.max(log_probs)
    log_norm_factor = p_max + np.log(np.sum(np.exp(log_probs - p_max)))
    norm_probs = np.exp(log_probs - log_norm_factor)
    results = np.random.multinomial(1, norm_probs)
    ind = results.nonzero()[0]
    if len(ind) != 0:
        return ind[0]
    return np.nan



