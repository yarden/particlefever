##
## Statistics utilities
##
import numpy as np

import scipy
import scipy.special
import scipy.misc
from scipy.special import gammaln

def log2sumexp(arr):
    max_val = arr.max()
    return np.log2(np.sum(2**(arr - max_val))) + max_val

def logsumexp(arr):
    return scipy.misc.logsumexp(arr)

def logC(alpha):
    """
    Following notation from:
    http://users.cecs.anu.edu.au/~ssanner/MLSS2010/Johnson1.pdf
    
    C(alpha) = [\prod_j=1^{m}\gamma(\alpha_j)] /
               \gamma(\sum_{j=1}^{m}\alpha_j)

    logC(alpha) = log(\prod_j=1^{m}\gamma(\alpha_j)]) - \
                  log(\gamma(\sum_{j=1}^{m}\alpha_j))
                  
                = \sum_{j=1}^{m}\gammaln(\alpha_j) - \
                  \gammaln(\sum_{j=1}^{m}\alpha_j))
    """
    # compute C function in log form:
    logC = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    return logC


if __name__ == "__main__":
    print "testing C function..."
    alpha = np.array([1, 1])
    counts = np.array([0, 0])
    logC(alpha)


