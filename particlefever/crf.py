##
## Chinese Restaurant Franchise
##
import os
import sys
import time

from collections import OrderedDict
import numpy as np

import particlefever

def init_ordered_dict(num_groups):
    """
    Initialize ordered dictionary indexed
    by a number from 0 to num_groups-1, with
    an empty list for each group value.
    """
    od = OrderedDict()
    for g in xrange(num_groups):
        od[g] = []
    return od

class CRF:
    """
    Chinese Restaurant Franchise. Following Teh et. al. (2006)
    notation.

    G_0 ~ DP(gamma, H)
    G_j ~ DP(alpha_0, G_0)
    """
    def __init__(self, gamma, alpha, num_groups, G_0):
        self.gamma = gamma
        self.alpha = alpha
        # number of groups (or restaurants)
        self.num_groups = num_groups
        # the global measure
        self.G_0 = G_0
        # global dish values (across all groups)
        self.all_dishes = []
        ##
        ## indices
        ##
        # t_ji = index of table associated with customer i
        #        in restaurant j
        self.cust_to_table_inds = init_ordered_dict(self.num_groups)
        # k_jt = index of dish (parameter) associated with table t
        #        in restaurant j
        self.dish_inds = init_ordered_dict(self.num_groups)
        ##
        ## counts
        ##
        # number of customers per table
        self.num_cust_per_table = init_ordered_dict(self.num_groups)
        # number of dishes 

    def __str__(self):
        num_total_customers = np.sum([np.sum(self.num_cust_per_table[g]) \
                                      for g in self.num_cust_per_table])
        num_total_tables = np.sum([len(self.num_cust_per_table[g]) \
                                   for g in self.num_cust_per_table])
        return "CRF(num_total_customers=%d, num_total_tables=%d, " \
               "num_groups=%d)" %(num_total_customers,
                                  num_total_tables,
                                  self.num_groups)

    def __repr__(self):
        return self.__str__()

    def sample_prior_table_assignment(self, g):
        """
        Sample new table assignment for a customer.

        Draw \theta_ji:
          \theta_ji | \theta_j,1, ..., \theta_j,i-1, \alpha, G_0
        """
        # get counts of customers at each table
        table_counts = self.num_cust_per_table[g]
        num_tables = len(table_counts)
        # by exchangeability, assume the new customer we're
        # drawing is the last one to come into restaurant.
        # So the new customer's number is total number of
        # customers plus 1
        num_customers = np.sum(table_counts)
        curr_cust = num_customers + 1
        # weight of table assignments to each of the existing
        # tables, plus a weight for being assigned
        # to a new table
        table_weights = table_counts + [self.alpha]
        table_weights /= (curr_cust - 1 + self.alpha)
        print "Table weights: ", table_weights
        # draw table assignment
        table_assignment = np.random.multinomial(1, table_weights).argmax()
        if table_assignment == (num_tables - 1):
            # customer joins new table, so draw new
            # parameter value
            new_param = self.sample_prior_table_param()
            self.
            self.dish_inds[g].append(new_param)
            # add customer to table counts
            self.num_cust_per_table[g].append(1)
            # update parameter counts across all tables
            # TODO: fill this in 
            pass
        else:
            # existing table was chosen, so update
            # number of customers seated in it
            self.num_cust_per_table[table_assignment] += 1

    def sample_prior_table_param(self):
        """
        Sample new table parameter.
        """
        # sample existing parameter in proportion to their
        # frequency and entirely new parameter in proportion
        # to \gamma, the concentration parameter of the global DP

        # first calculate the frequency of dishes across groups
        dish_frequencies = 
        
    def sample_prior(self):
        """
        Sample from prior.
        """
        table_assignments = self.sample_table_assignments()
        print "table assignments: "
        print table_assignments


if __name__ == "__main__":
    gamma = 1
    alpha = 1
    num_groups = 3
    G_0 = None
    crf_obj = CRF(gamma, alpha, num_groups, G_0)
    print crf_obj

    
