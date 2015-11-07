##
## Node in a PGM
##
import os
import sys
import time

import numpy as np

import particlefever

class Node(object):
    """
    Node in a graphical model.
    """
    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents
        self.children = children

        def logp(self):
            """
            log probability.
            """
            return None

        def rand(self):
            """
            Sample value for node.
            """
            return None
        

class Bernoulli(Node):
    def __init__(self, name, p, parents=[], children=[]):
        Node.__init__(name, parents=parents, children=children)
        self.p = p

    def logp(self):
        """
        log probability of Bernoulli.
        """
        pass

    def rand(self):
        """
        Sample from Bernoulli.
        """
        pass
