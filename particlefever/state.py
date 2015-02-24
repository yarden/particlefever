##
## State representation (pure Python)
##
import os
import sys
import time

import particlefever

class State:
    def __init__(self, name):
        self.name = name
        self.parents = []

    def sample_value(self):
        """
        Sample value for state.
        """
        pass



