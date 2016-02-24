##
## Brute force calculation of posterior by gridding.
##
import os
import sys
import time
import itertools

import numpy as np

import particlefever
import particlefever.bayes_hmm as bayes_hmm

import cPickle as pickle
import shelve

class GridDiscreteBayesHMM:
    """
    Gridder for a discrete Bayesian HMM.
    """
    def __init__(self, hmm):
        self.hmm = hmm

    def solve(self, data, output_fname):
        """
        Solve by gridding.
        """
        num_computations = 0
        seq_len = data.shape[0]
        print "solving HMM by gridding (%d data points)" %(seq_len)
        t1 = time.time()
        if os.path.isfile(output_fname):
            raise Exception, "%s exists." %(output_fname)
        out_file = open(output_fname, "w")
        for state_point in make_space(data):
            #print "state point: ", state_point
            self.hmm.trans_mat = state_point[0]
            self.hmm.out_mat = state_point[1]
            self.hmm.hidden_state_trajectory = state_point[2]
            log_score = bayes_hmm.log_score_joint(self.hmm)
            hidden_traj_str = "".join(map(str, self.hmm.hidden_state_trajectory))
            trans_mat_str = np.array_str(self.hmm.trans_mat,
                                        precision=4).replace("\n", ",")
            out_mat_str = np.array_str(self.hmm.out_mat,
                                       precision=4).replace("\n", ",")
            entry = "%s\t%s\t%s\t%.4f\n" %(hidden_traj_str,
                                           trans_mat_str,
                                           out_mat_str,
                                           log_score)
            print entry
            out_file.write(entry)
            # index results by hidden state trajectory
            if (num_computations % 100000) == 0:
                print "through %d configurations" %(num_computations)
            num_computations += 1
        out_file.close()
        t2 = time.time()
        print "made %d computations in %.2f mins" %(num_computations,
                                                    (t2 - t1)/60.)

def make_space(data):
    """
    Generate space of HMM.
    """
    seq_len = data.shape[0]
    num_hidden_trajectories = np.power(2, seq_len)
    # calculate number of computations to be made
#    num_computations = \
#      np.log10(num_mats) + np.log10(num_mats) + np.log10(num_hidden_trajectories)
    for hidden_trajectory in get_hidden_state_space(seq_len):
        for trans_mat in grid_prob_matrix():
            for emit_mat in grid_prob_matrix():
                yield (trans_mat, emit_mat, hidden_trajectory)

    
def get_hidden_state_space(seq_len, num_states=2):
    if num_states != 2:
        raise Exception, "Only implemented for binary states."
    for seq in itertools.product("01", repeat=seq_len):
        yield np.array(map(int, seq))
    
def grid_prob_matrix(shape=(2,2), num_prob_bins=20):
    """
    Returns:
      An array of np.arrays and a bin step
    """
    if not (shape[0] == shape[1] == 2):
        raise Exception, "Only defined for 2x2 matrices."
    num_mats = np.power(num_prob_bins, 2)
    mats = np.zeros((num_mats, shape[0], shape[1]))
    near_zero = np.power(10., -2)
    near_one = 1 - near_zero
    prob_bins = np.linspace(near_zero, near_one, num_prob_bins)
    n = 0
    t1 = time.time()
    for row1 in prob_bins:
        for row2 in prob_bins:
            curr_mat = np.array([[row1, 1 - row1],
                                 [row2, 1 - row2]])
            yield curr_mat

def save_state_space(output_fname):
    """
    Save state space.
    """
    pass

def main():
    num_pairs = 5
    data = np.array([0, 1] * num_pairs)
    init_probs = np.array([0.5, 0.5])
    # settings for hyperparameters
    trans_alpha = 1.
    trans_mat_hyperparams = np.ones((2, 2))
    trans_mat_hyperparams *= trans_alpha 
    out_alpha = 1.
    out_mat_hyperparams = np.ones((2, 2))
    out_mat_hyperparams *= out_alpha
    init_state_hyperparams = np.ones(2)
    init_state_hyperparams *= 1. 
    # make HMM
    hmm_obj = \
      bayes_hmm.DiscreteBayesHMM(2, 2,
                                 trans_mat_hyperparams=trans_mat_hyperparams,
                                 out_mat_hyperparams=out_mat_hyperparams,
                                 init_state_hyperparams=init_state_hyperparams)
    hmm_obj.init_probs = init_probs
    hmm_obj.outputs = data
    gridder = GridDiscreteBayesHMM(hmm_obj)
    gridder.solve(data, "./hmm_result")
    
    
if __name__ == "__main__":
    main()
