##
## Profile code
##
import os
import sys
import time

import numpy as np

import cProfile

def run_hmm():
    trans_mat_hyperparams = np.ones((2, 2))
    trans_mat_hyperparams *= 1.
    # put peaky prior on outputs
    out_mat_hyperparams = np.ones((2, 2))
    out_mat_hyperparams *= 0.1
    simple_hmm = \
      bayes_hmm.DiscreteBayesHMM(2, 2,
                                 trans_mat_hyperparams=trans_mat_hyperparams,
                                 out_mat_hyperparams=out_mat_hyperparams)

def main():
    cProfile.run('run_hmm()')

if __name__ == "__main__":
    main()
