##
## Dynamic Bayes net representation
##
import os
import sys
import time

import pymc

import copy

import particlefever
import particlefever.pgm as pgm

class DBN:
    """
    Dynamic Bayes Net.
    """
    def __init__(self, name=""):
        self.name = name
        # Set of time dependencies. These specify the core
        # of the model.
        self.time_deps = {}
        # Current time
        self.curr_time = 0
        # Models at time t-1,t-2,...,t-N
        # encoded as tuples: [(t-1, m1), (t-2, m2), etc...]
        self.prev_models = []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DBN(name=%s, num_times=%d)" \
               %(self.name, self.num_times)

    @property
    def num_times(self):
        """
        Return number of time slices in model.
        """
        return len(self.time_models)

    def add_time(self):
        """
        Add another time slice to the model. This means
        replicating the initial time model to the next time
        step. NOTE: this means that we're not allowing the number of
        variables to change across time. Each time slice must
        have the exact same variables as the initial model.
        """
        pass

    def add_time_dep_logp(self, var_name, curr_time, time_logp):
        """
        Specify time dependent logp between a node and a set
        of previous nodes, i.e.

        depends(rain, t, time_logp)

        where time_dep_func is a function, arguments are the
        list of variables and time indices that rain @ time t
        depends on.

        e.g.

        def time_dep_logp(vars, [('rain', t-1), ('umbrella', t-2)]):
          # specify here the probability that rain takes on
          # a value at time t, given 'rain' at t-1 and 'umbrella'
          # at time t-2
          return p

        Args:
        - curr_time: time point for which probability is specified
        - time_deps: list of time dependencies; variables
        """
        if (varname, curr_time) in self.time_deps:
            raise Exception, "Already have time logp for %s @ t=%s" \
                  %(varname, str(curr_time))
        self.time_deps[(varname, curr_time)] = time_logp

    def get_markov_blanket(self, node):
        """
        Return Markov blanket of node. Takes time dependencies
        into account.
        
        A node's Markov blanket is the node's Markov blanket in the current
        time point, plus the nodes that affect it back in time, across
        in the set of relevant previous time points
        """
        # Get current time's Markov blanket
        markov_blanket = pgm.get_markov_blanket(node, self.curr_model)
        # Look at the relevant set of previous time points to get
        # the node's dependencies
        return markov_blanket

    def set_curr_time(self, t):
        """
        Set current time.
        """
        if t > self.num_times:
            raise Exception, "%d not valid time (only %d time slices)" \
                  %(t, self.num_times)
        self.curr_time = t

    def forward_sample(self):
        """
        Generate a forward sample from the model. This is a 
        setting of all the nodes given the previous time step.
        """
        # If we're in the first time step, just generate a sample
        # from all the variables
        next_model = None
        if self.curr_time == 0:
            next_model = self.curr_model.draw_from_prior()
        else:
            # If we're not in the first time step, generate a sample
            # conditional on the relevant number of previous samples
            ##
            ## for each node in the current model
            ##   - get the node's markov blanket
            ##       the markov blanket includes previous time dependencies
            ##   - sample value for node conditioned on its Markov blanket
            for node in self.curr_model.nodes:
                node_blanket = self.get_markov_blanket(node)
                print "node: ", pgm.get_node_name(node)
                print "Markov blanket: ", node_blanket
                print "-"*50
                # Sample value for node conditioned on Markov blanket
                # ...
        self.advance_time(next_model)

    def unroll_to_pgm(self, model_func, num_steps):
        """
        Unroll DBN to PGM.
        """
        for n in range(num_steps):
            for node in self.nodes:
                pass

    # def advance_time(self, next_model):
    #     """
    #     Move forward in time. Set 'next_model' to be
    #     the current time model.
    #     """
    #     self.curr_time_model = next_model
    #     # Advance time index forward
    #     self.curr_time += 1

    # def get_max_time_dep(self, node):
    #     """
    #     Get maximum time dependency for a given node.
    #     """
    #     # iterate through the time conditionals and calculate
    #     # the maximum dependency
    #     # for time_cond in self.time_conditionals:
    #     #   # calculate maximum dependency (i.e. look at size of 'prev_nodes')
    #     #   # record it
    #     # take max here
    #     pass

    # def unroll_to_pgm(self, num_steps):
    #     """
    #     Unroll DBN to a PGM. Expand out the DBN to a PGM.
    #     """
    #     curr_pgm = copy.copy(self.init_model.value)
    #     unrolled_pgm = None
    #     pgms = []
    #     for t in range(num_steps):
    #         new_pgm = pgm.rename_pgm(curr_pgm, "_%d" %(t))
    #         pgms.append(new_pgm)
    #     print "pgms: "
    #     for p in pgms:
    #         print p
    #         for n in p.variables:
    #             print n, n.__name__
    #         print "-" * 15
    #     print "init model: "
    #     print self.init_model
    #     for t in self.init_model.variables:
    #         print t, t.__name__
    #     return pgms
        
            
            
            

#@pymc.stochastic()


