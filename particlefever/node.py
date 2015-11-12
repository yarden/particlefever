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
    def __init__(self, name="", doc="", parents={}, children=[]):
        self.name = name
        self.doc = doc
        self.parents = parents
        self.children = children

    def __repr__(self):
        return self.__init__()

    def __str__(self):
        return "Node(%s)" %(self.name)


class Stochastic(Node):
    """
    Stochastic node.
    """
    def __init__(self, name="", doc="", parents={}, children=[],
                 logp=None, random=None):
        Node.__init__(self, name, doc=doc, parents=parents, children=children)
        self.logp = logp
        self.random = random

    def get_logp(self):
        return self.logp
    
    def __str__(self):
        return "Stochastic(%s)" %(self.name)

    
class Bernoulli(Stochastic):
    def __init__(self, name, p, parents={}, children=[], logp=None):
        Node.__init__(self, name=name, parents=parents, children=children)
        self.p = p
        self.logp = logp

    def rand(self):
        """
        Sample from Bernoulli.
        """
        pass

    def __str__(self):
        return "Bernoulli(%s)" %(self.name)
        
