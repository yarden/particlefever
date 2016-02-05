##
## Rain-Umbrella
##
import pymc
import numpy as np

import particlefever
import particlefever.dbn as dbn

rain = pymc.Bernoulli("rain", 0.5)

def copy_model(model):
    """
    Copy a PyMC model.

    Attributes to copy:
    - deterministics
    - stochastics (with observed=False)
    - data (stochastic variables with observed=True)
    - variables
    - potentials
    - containers
    - nodes
    - all_objects
    - status: Not useful for the Model base class, but may be used by subclasses.

    The following attributes only exist after the appropriate method is called:
    - moral_neighbors: The edges of the moralized graph. A dictionary, keyed by stochastic variable,
    whose values are sets of stochastic variables. Edges exist between the key variable and all variables
    in the value. Created by method _moralize.
    - extended_children: The extended children of self's stochastic variables. See the docstring of
    extend_children. This is a dictionary keyed by stochastic variable.
    - generations: A list of sets of stochastic variables. The members of each element only have parents in
    previous elements. Created by method find_generations.    
    """
    new_model = pymc.Model({})
    pass

    

def umbrella_logp(value, rain):
    """
    Umbrella node. Initial value is False.

    P(umbrella=True | rain=True)  = 0.9
    P(umbrella=True | rain=False) = 0.2
    """
    p_umb_given_rain = 0.9
    p_umb_given_no_rain = 0.2
    if rain:
        logp = pymc.bernoulli_like(value, p_umb_given_rain)
    else:
        logp = pymc.bernoulli_like(value, p_umb_given_no_rain)
    return logp


def umbrella_random(rain):
    return (np.random.rand() <= np.exp(umbrella_logp(True, rain)))

# Observed node
umbrella = pymc.Stochastic(name="umbrella",
                           doc="umbrella var",
                           parents={"rain": rain},
                           logp=umbrella_logp,
                           observed=True,
                           value=True)

#rain_model = pymc.Model([rain, umbrella])


def copy_node(node):
    """
    Copy PyMC node.
    """
    print type(node)
    return node


import pymc
import copy
rain = pymc.Bernoulli("rain", 0.5)
model = pymc.Model([rain])
# Get a variable from a model and try to copy it
var = next(iter(model.variables))
new_node = copy_node(var)
print new_node.__name__
print new_node._parents
print "New node name after replacement"
new_node.__name__ = "foo"
print new_node.__name__
print "Is the original node name modified?"
print var.__name__
print var._parents
raise Exception, "test"

# This is not a copy of model!
new_model = model.value
# This is not a copy of model either
new_model = copy.deepcopy(model.value)
new_rain = new_model.get_node("rain")
# This modified original model
new_rain.__name__ = "new_rain"
print "We modified original model: "
print model.variables
print "Before replacement: "
# How to modify the variable of a model?
for n in new_model.variables:
    print n, n.__name__
new_model.replace(None, "rain", new_model.get_node("rain"))
print "After replacement: "
for n in new_model.variables:
    print n, n.__name__
