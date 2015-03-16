##
## Tools for working with probabilistic graphical models (PGMs)
##
import pymc
import numpy as np

def get_markov_blanket(node, model, exclude_self=True):
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
    if exclude_self:
        # Remove the node from its own Markov blanket if it
        # is in it
        markov_blanket = [n for n in markov_blanket \
                          if n is not node]
    return markov_blanket

def is_variable_node(node):
    """
    Check if a given node is a variable node, i.e.
    a Stochastic, Deterministic, or Potential node.
    """
    if isinstance(node, pymc.Variable):
        return True
    return False

def get_node_name(node):
    """
    Return node's name. For some reason, name is not exposed
    in pymc variables (e.g. Stochastic nodes.)
    """
    return node.__name__
    



