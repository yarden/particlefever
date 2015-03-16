##
## Tools for working with probabilistic graphical models (PGMs)
##
import pymc
import numpy as np

def get_markov_blanket(node, model):
    """
    Get Markov model of a node in pymc model. The Markov blanket
    of a node is the node's set of children, parents, and
    children's parents.
    """
    # Collect children
    children = list(node.children)
    # Collect parents
    parents = []
    for parent_name in node.parents:
        parent_node = node.parents[parent_name]
        if parent_node in model.nodes:
            parents.append(parent_node)
    # Collect children's parents
    parents_of_children = []
    for child in children:
        for pc_name in child.parents:
            pc_node = child.parents[pc_name]
            parents_of_children.append(pc_node)
    markov_blanket = children + parents + parents_of_children
    return markov_blanket
    
    



