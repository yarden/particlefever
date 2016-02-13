##
## Profile code
##
import os
import sys
import time

import numpy as np
import cProfile

import particlefever
import particlefever.bayes_hmm as bayes_hmm
import particlefever.sampler as sampler

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
    data = np.array([0, 1]*20)
    print "data: ", data
    gibbs_obj = sampler.DiscreteBayesHMMGibbs(simple_hmm)
    gibbs_obj.sample(data)


def profile_discrete_hmm():
    print "Profiling discrete HMM"
    cProfile.run('run_hmm()')
        

def main():
    profile_discrete_hmm()

if __name__ == "__main__":
    main()
