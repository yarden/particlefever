import numpy as np


def p_decorate(func):
    def func_wrapper(*args, **kwargs):
        print "args: ", args
        print func.__code__.co_varnames
        print "<p>{0}</p>".format(func(*args, **kwargs))
    return func_wrapper

@p_decorate
def logp(var_name, t, other_vars):
    """
    pass.
    """
    print "var name: ", var_name


print "results: "
print logp("rain", "t", [("rain", "t-1")])


#@time_decorate("t1")
#def logp(var_name, other_vars)

def time_logp("rain_t", "rain_t-1"):
    pass
    
    



















