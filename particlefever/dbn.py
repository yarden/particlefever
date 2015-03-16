##
## Dynamic Bayes net representation
##
import os
import sys
import time

import pymc

import particlefever
import particlefever.pgm as pgm


class DBN:
    """
    Dynamic Bayes net. 
    """
    def __init__(self, init_model, name=""):
        self.init_model = init_model
        self.name = name
        # Number of time slices in model
        # starts with 1, because we always have the initial model
        self.num_times = 1
        # Set of time conditionals. These specify the core
        # of the model.
        self.time_conditionals = {}
        # Model at current time
        ##### NOTE: Do we need to make a copy of init_model?
        self.curr_model = init_model
        self.curr_time = 0
        self.time_models = []
        # Models at time t-1,t-2,...,t-N
        # encoded as tuples: [(t-1, m1), (t-2, m2), etc...]
        self.prev_models = []

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DBN(name=%s, num_times=%d)" \
               %(self.name, self.num_times)

    def init_time(self):
        """
        Initialize the time.
        """
        pass

    def add_time(self):
        """
        Add another time slice to the model. This means
        replicating the initial time model to the next time
        step. NOTE: this means that we're not allowing the number of
        variables to change across time. Each time slice must
        have the exact same variables as the initial model.
        """
        self.time_models.append(self.init_model)
        self.num_times += 1
        print "Advanced time (%d time slices now)" %(self.num_times)

    def add_time_conditional(self, var_to_cond_func):
        """
        Specify conditional relationship between a node and a set
        of previous nodes.

        Args:
        - var_to_cond_func: mapping from variable name to conditional
          function, e.g. {"X": func} which says that "X" depends on
          'func'.
        """
        for var in var_to_cond_func:
            if var in self.time_conditionals:
                # Each variable can only have a single time conditional
                raise Exception, "Variable %s already has time conditional" \
                                 %(var)
            self.time_conditionals[var] = var_to_cond_func[var]

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
        return []

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
            print "NODES: ", self.curr_model.nodes, type(self.curr_model.nodes)
            for node in self.curr_model.nodes:
                node_blanket = self.get_markov_blanket(node)
                # Sample value for node conditioned on Markov blanket
                print "sampling value conditioned on mblanket: ", node_blanket
                # ...
#            for node in self.model:
#                pass
        # Advance our sample
        self.advance_time(next_model)

    def advance_time(self, next_model):
        """
        Move forward in time. Set 'next_model' to be
        the current time model.
        """
        self.curr_time_model = next_model
        # Advance time index forward
        self.curr_time += 1

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


