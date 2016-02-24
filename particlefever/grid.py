##
## Brute force calculation of posterior by gridding.
##
import os
import sys
import time

import numpy as np

import particlefever
import particlefever.bayes_hmm as bayes_hmm

class GridDiscreteBayesHMM:
    """
    Gridder for a discrete Bayesian HMM.
    """
    def __init__(self, hmm):
        self.hmm = hmm

    def solve(self):
        """
        Solve by gridding.
        """
        pass

def grid_prob_matrix(shape=(2,2), num_prob_bins=20):
    """
    Returns:
      An array of np.arrays and a bin step
    """
    if not (shape[0] == shape[1] == 2):
        raise Exception, "Only defined for 2x2 matrices."
    num_mats = np.power(num_prob_bins, 2)
    mats = np.zeros((num_mats, shape[0], shape[1]))
    prob_bins = np.linspace(0., 1., num_prob_bins)
    n = 0
    t1 = time.time()
    for row1 in prob_bins:
        for row2 in prob_bins:
            curr_mat = np.array([[row1, 1 - row1],
                                 [row2, 1 - row2]])
            mats[n, :] = curr_mat
            n += 1
    t2 = time.time()
    print "generated %d matrices in %.2f mins" %(n, (t2 - t1)/60.)
    assert (n == num_mats), "Did not generated all matrices."
    return mats

def main():
    #hmm = bayes_hmm.DiscreteBayesHMM(2, 2)
    #gridder = GridDiscreteBayesHMM(hmm)
    import itertools
    print "beginning iteration..."
    mats = grid_prob_matrix()
    
if __name__ == "__main__":
    main()
