##
## Dirichlet process
##
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

def dir_process_norm_stickbreak(base_measure, alpha, num_draws=50):
    """
    Draw values from Dirichlet process using stick breaking.
    """
    # locations: i.i.d. from base measure
    locs = base_measure(loc=1, scale=0.5, size=num_draws)
    # betas: i.i.d. probabilities
    betas = np.random.beta(1, alpha, size=num_draws)
    remaining_lens = np.cumprod(float(1) - betas)
    weights = remaining_lens * betas
    print "weights: ", weights
    indices = [np.random.multinomial(1, weights).argmax() \
               for n in xrange(num_draws)]
    print indices
    values = locs[indices]
    return values, weights


if __name__ == "__main__":
    alphas = [0.5, 3, 30]
    num_draws = 10
    plt.figure(figsize=(6, 3))
    fig = plt.gcf()
    fig.suptitle("Stick breaking Dirichlet process",
                 fontsize=14)
    sns.set_style("ticks")
    for n, alpha in enumerate(alphas):
        plt.subplot(1, len(alphas), n + 1)
        values, weights = \
          dir_process_norm_stickbreak(np.random.normal, alpha,
                                      num_draws=num_draws)
        plt.bar(range(len(weights)), weights,
                color="k")
        plt.xlabel("Draw")
        plt.ylabel("Weight")
        plt.title(r"$\alpha = %.2f$" %(alpha))
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()
