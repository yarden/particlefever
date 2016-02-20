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
import particlefever.switch_ssm as switch_ssm
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


def run_ssm():
    data = np.array([0, 1] * 50)# + [1, 1] * 10)
    ssm = switch_ssm.DiscreteSwitchSSM(2, 2)
    gibbs_obj = sampler.DiscreteSwitchSSMGibbs(ssm)
    gibbs_obj.sample(data, num_iters=2000, burn_in=100)
    num_preds = 50
    pred_probs = switch_ssm.get_predictions(gibbs_obj.samples, num_preds)


def profile_discrete_hmm():
    print "profiling discrete HMM"
    cProfile.run('run_hmm()')


def profile_discrete_ssm():
    print "profiling discrete SSM"
    cProfile.run('run_ssm()', "restats")
    import pstats
    p = pstats.Stats("restats")
    p.sort_stats('cumulative').print_stats(50)
        

def main():
    profile_discrete_ssm()

if __name__ == "__main__":
    main()
