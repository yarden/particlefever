##
## Bayesian HMM
##
import os
import sys
import time

import numpy as np

class DiscreteBayesHMM:
    """
    Discrete Bayesian HMM.
    """
    def __init__(self, trans_mat, out_mat,
                 prior_trans_mat=None, prior_out_mat=None):
        self.init_probs = None
        self.trans_mat = trans_mat
        self.out_mat = out_mat
        self.prior_trans_mat = prior_trans_mat
        self.prior_out_mat = prior_out_mat

    def initialize(self):
        """
        Initialize to random model.
        """
        pass
