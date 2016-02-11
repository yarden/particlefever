##
## Chinese Restaurant Franchise
##
import os
import sys
import time

from collections import OrderedDict
import numpy as np

import particlefever

def init_ordered_dict(num_groups, dtype=np.int32):
    """
    Initialize ordered dictionary indexed
    by a number from 0 to num_groups-1, with
    an empty list for each group value.
    """
    od = OrderedDict()
    for g in xrange(num_groups):
        od[g] = np.array([], dtype=dtype)
    return od

class CRF:
    """
    Chinese Restaurant Franchise. Following Teh et. al. (2006)
    notation.

    G_0 ~ DP(gamma, H)
    G_j ~ DP(alpha_0, G_0)
    """
    def __init__(self, gamma, alpha, num_groups, G_0, H):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        # number of groups (or restaurants)
        self.num_groups = num_groups
        # the global DP
        self.G_0 = G_0
        # the global DP's base measure
        self.H = H
        # global dish values (across all groups)
        self.all_dishes = np.array([])
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
        # number of dishes across all groups
        self.all_dish_counts = np.array([], dtype=np.int32)
        # total number of occupied tables
        self.total_occupied_tables = 0

    def __str__(self):
        num_total_customers = np.sum([np.sum(self.num_cust_per_table[g]) \
                                      for g in self.num_cust_per_table])
        num_total_tables = self.total_occupied_tables
        return "CRF(num_total_customers=%d, num_total_tables=%d, " \
               "num_groups=%d,\ncust_per_table=%s,\ndish_inds=%s)" \
               %(num_total_customers,
                 num_total_tables,
                 self.num_groups,
                 str(self.num_cust_per_table),
                 str(self.dish_inds))

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
        table_weights = np.array(list(table_counts) + [self.alpha])
        table_weights /= (curr_cust - 1 + self.alpha)
        # draw table assignment
        table_assignment = np.random.multinomial(1, table_weights).argmax()
        if (table_assignment == num_tables) or (num_tables == 0):
            # customer joins new table, so draw new
            # parameter value
            new_dish = self.sample_prior_dish()
            self.dish_inds[g] = np.append(self.dish_inds[g], new_dish)
            # add customer to table counts
            self.num_cust_per_table[g] = np.append(self.num_cust_per_table[g], 1)
            self.total_occupied_tables += 1
        else:
            # existing table was chosen, so update
            # number of customers seated in it
            self.num_cust_per_table[g][table_assignment] += 1
        return table_assignment 

    def sample_prior_dish(self):
        """
        Sample new dish parameter.
        """
        # sample existing parameter in proportion to their
        # frequency or entirely new parameter in proportion
        # to \gamma, the concentration parameter of the global DP
        dish_weights = np.array(list(self.all_dish_counts) + [self.gamma])
        dish_weights /= (self.total_occupied_tables + self.gamma)
        dish_assignment = np.random.multinomial(1, dish_weights).argmax()
        num_dishes = len(self.all_dishes)
        if (dish_assignment == num_dishes) or (num_dishes == 0):
            # draw new dish parameter
            new_dish = self.H()
            self.all_dishes = np.append(self.all_dishes, new_dish)
            # update dish frequency popularity
            self.all_dish_counts = np.append(self.all_dish_counts, 1)
        else:
            # we've chosen an existing dish
            self.all_dish_counts[dish_assignment] += 1
            # update dish frequency popularity
        return dish_assignment
        
    def sample_prior(self, g):
        """
        Sample from prior.
        """
        table_assignments = self.sample_prior_table_assignment(g)


def plot_crf(crf_obj):
    """
    Plot CRF.
    """
    for g in xrange(crf_obj.num_groups):
        plt.subplot(crf_obj.num_groups, 1, g + 1)


if __name__ == "__main__":
    gamma = 1.
    alpha = 1.
    num_groups = 3
    G_0 = None
#    np.random.seed(5000)
    H = lambda: np.random.normal()
    crf_obj = CRF(gamma, alpha, num_groups, G_0, H)
    num_samples = 50
    for n in xrange(num_samples):
        for g in xrange(num_groups):
            crf_obj.sample_prior(g)
    print crf_obj
            

    
