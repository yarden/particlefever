##
## plotting
##
import os
import sys
import time

import copy

import numpy as np

import particlefever
import particlefever.bayes_hmm as bayes_hmm

import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

def plot_logscores_vary_alpha():
    num_bins = 1000
    # define constants
    zero = np.power(10., -5)
    one = 1. - zero
    alpha_start = 0.1
    alpha_end = 10.
    log_scores = []
    trans_alpha = 1.
    trans_mat_hyperparams = np.ones((2, 2))
    trans_mat_hyperparams *= trans_alpha
    num_pairs = 20
    data = np.array([0, 1] * num_pairs)
    init_probs = np.array([0.5, 0.5])
    out_alpha_vals = np.linspace(alpha_start, alpha_end, num_bins)
    for out_alpha in out_alpha_vals:
        out_mat_hyperparams = np.ones((2, 2))
        out_mat_hyperparams *= out_alpha
        # score models
        # periodic HMM
        hmm1 = \
          bayes_hmm.DiscreteBayesHMM(2, 2,
                                     trans_mat_hyperparams=trans_mat_hyperparams,
                                     out_mat_hyperparams=out_mat_hyperparams)
        hmm1.init_probs = init_probs
        hmm1.trans_mat = np.array([[zero, one],
                                   [one, zero]])
        hmm1.out_mat = np.array([[one, zero],
                                 [zero, one]])
        hmm1.hidden_state_trajectory = data
        hmm1.outputs = data
        # fair coin HMM
        hmm2 = copy.deepcopy(hmm1)
        hmm2.init_probs = init_probs
        hmm2.trans_mat = np.array([[one, zero],
                                   [zero, one]])
        hmm2.out_mat = np.array([[0.5, 0.5],
                                 [0.5, 0.5]])
        hmm2.hidden_state_trajectory = np.array([0] * len(data))
        hmm2.outputs = data
        log_score1 = bayes_hmm.log_score_joint(hmm1)
        log_score2 = bayes_hmm.log_score_joint(hmm2)
        ratio = (log_score1 - log_score2) / np.log2(np.e)
        log_scores.append([out_alpha, log_score1, log_score2, ratio])
    # plot results
    log_scores = np.array(log_scores)
    plt.figure()
    sns.set_style("ticks")
    plt.ylabel(r"Log scores ratio, periodic vs. fair hidden states ($\log_{2}$)")
    plt.xlabel(r"Emission matrix hyperparameter, $\alpha_E$")
    plt.title(r"Hidden state estimation ($\alpha_T = %.1f$)" \
              %(trans_alpha))
    plt.axhline(y=0, color="#999999", label="Equally scoring modes")
    plt.plot(log_scores[:, 0], log_scores[:, 3], color="k")
    plt.legend(loc="upper right")
    # find point where ratio is closest to 0
    sns.despine(trim=True, offset=2)
    plt.savefig("./logscores_periodic_vs_fair_vary_alpha.pdf")
    plt.show()

def main():
    plot_logscores_vary_alpha()

if __name__ == "__main__":
    main()
