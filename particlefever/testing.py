##
## Testing
##
import os
import sys
import time
import unittest
import numpy as np
import scipy
import scipy.stats 

import particlefever
import particlefever.math_utils as math_utils

class TestScoring(unittest.TestCase):
    """
    Test scoring functions.
    """
    def test_multinomial_score(self):
        """
        Test multinomial scoring function
        """
        # test multinomial scoring function
        n = 10
        val_multi = math_utils.multinomial_pmf([5, 5], [0.5, 0.5])
        val_binom = scipy.stats.binom.pmf(5, 10, 0.5)
        assert (np.isclose(val_multi, val_binom)), \
          "multinomial and binomial pmfs give different answers."
        # test scoring in log space
        val_multi = math_utils.multinomial_logpmf([5, 5], [0.5, 0.5])
        val_binom = scipy.stats.binom.logpmf(5, 10, 0.5)
        assert (np.isclose(val_multi, val_binom)), \
          "multinomial and binomial logpmfs give different answers."
        
        

if __name__ == "__main__":
    unittest.main()

