##
## Hierarchical Dirichlet process
##
import os
import sys
import time

import numpy as np

class HierarchicalDP:
    """
    Hierarchical Dirichlet process.

    A Dirichlet process that has another Dirichlet process
    as a base measure.

    # sample weights from a Dirichlet process (parent DP)
    beta ~ DP(gamma, H), where H is Dirichlet
      [equivalently: beta ~ Stick(gamma) or beta ~ GEM(gamma)]
    
    # sample weights from a Dirichlet process with
    # parent DP as base measure
    alpha ~ DP(alpha, beta)
    """
    def __init__(self, gamma, alpha):
        """
        Args:
        -----
        - gamma: prior on stick breaking
        """
        self.gamma = gamma
        self.alpha = alpha
        self.parent_dp = None

    def sample(self, num_samples):
        """
        Sample from hierarchical DP.
        """
        # if we don't already have a parent DP,
        # start one
        if self.parent_dp is None:
            self.parent_dp = 
        for n in xrange(num_samples):
            pass
            

