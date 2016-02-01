##
## Sampler
##

class Sampler(object):
    def __init__(self):
        pass

class Gibbs(Sampler):
    """
    Gibbs sampler for discrete Bayesian HMM.
    """
    def __init__(self, model):
        self.model = model
        self.hidden_trajectories = []

    def sample(self, num_iters=1000, burn_in=100, lag=2):
        """
        Run posterior sampling.
        """
        num_nodes = self.model.num_hidden_nodes
        hidden_trajectories = np.zeros(num_iters, dtype=np.int32)
        for n_iter in xrange(num_iters):
            for n_var in xrange(num_nodes):
                pass
    
