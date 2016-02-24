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
        
        

def grid_prob_matrix(shape, num_prob_bins=200):
    """
    Returns:
      An array of np.arrays and a bin step
    """
#    prob_bins = np.linspace()
#    matrices = np.array()
    for row in shape[0]:
        for col in shape[1]:
            pass
             

def main():
    #hmm = bayes_hmm.DiscreteBayesHMM(2, 2)
    #gridder = GridDiscreteBayesHMM(hmm)
    import itertools
    t1 = time.time()
    for c in itertools.count((100**4) * (100**4) * (2**10)):
        pass
    t2 = time.time()
    print "count took %.2f mins" %(t2 - t1)
        

if __name__ == "__main__":
    main()
