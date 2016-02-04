##
## Utilities related to Markov models
##
import numpy as np

def sample_trans_mat_from_dirch(hyperparams):
    """
    Sample a transition matrix from a symmetric Dirichlet prior.
    Returns an np.matrix.

    Args:
    -----
    - hyperparameters: an np.array vector of parameters for independent
    symmetric Dirichlet distributions, where i-ith position is the
    parameter on the Dirichlet for the i-th row.
    """
    num_rows = hyperparams.shape[0]
    trans_mat = np.zeros((num_rows, num_rows))
    for n in xrange(num_rows):
        # sample values for current row
        # assume a symmetric Dirichlet, so each row is
        # parameterized as Dirch([alpha, alpha, ...])
        # (all parameters are identical)
        row = np.random.dirichlet([float(hyperparams[n])] * num_rows)
        trans_mat[n, :] = row
    trans_mat = np.matrix(trans_mat)
    return trans_mat


if __name__ == "__main__":
    print "Sampling transition matrices: "
    print "even Dirichlet hyperprior:"
    hyperparams = np.array([10, 10])
    trans_mat = sample_trans_mat_from_dirch(hyperparams)
    print trans_mat
    print "sparse Dirichlet hyperprior:"
    hyperparams = np.array([0.1, 0.1])
    trans_mat = sample_trans_mat_from_dirch(hyperparams)
    print trans_mat
    
