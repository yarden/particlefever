##
## Dynamic Bayes net representation
##
import os
import sys
import time

import pymc

import particlefever


class DBN:
    """
    Dynamic Bayes net. 
    """
    def __init__(self, name=""):
        self.name = name
        # Set of time conditionals. These specify the core
        # of the model.
        self.time_conditionals = []
        # Model at time t 
        self.curr_model = None
        # Models at time t-1,t-2,...,t-N
        self.prev_models = []


    def init_time(self):
        """
        Initialize the time.
        """
        pass


    def add_conditional(self, conditional_func, t_curr_node, t_prev_nodes):
        """
        Specify conditional relationship between a node and a set
        of previous nodes.

        Args:
        - conditional_func: conditional function. Takes a current
          node and a set of previous nodes and produces the conditional
          distribution.
        - t_curr_node: current time node
        - t_prev_nodes: previous nodes
        """
        pass


    def forward_sample(self):
        """
        Generate a forward sample from the model. This is a
        setting of all the nodes given the previous time step.
        """
        # If we're in the first time step, just generate a sample
        # from all the variables
        #if at first time -> get initial sample

        # If we're not in the first time step, generate a sample
        # conditional on the relevant number of previous samples
        pass


    def get_max_time_dep(self):
        """
        Get maximum time dependency for the network. This is the
        maximum number of time steps that affect any node in the
        network. I.e., if a node in the current time step depends on
        the last 5 time steps, and all the rest depend on time steps < 5,
        then the maximum time dependency is 5.
        """
        # iterate through the time conditionals and calculate
        # the maximum dependency
        # for time_cond in self.time_conditionals:
        #   # calculcate maximum dependency (i.e. look at size of 'prev_nodes')
        #   # record it
        # take max here
        pass
        

#@pymc.stochastic()


