##
## Utilities related to Markov models
##
import numpy as np
from itertools import chain, product

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


def cond_probs_np(sequences):
    """
    Calculate 1-step transitional probabilities from sequences.
    Return dictionary mapping transitions to probabilities.
    """
    distinct = set(chain.from_iterable(sequences))
    n = len(distinct)
    # so that they will fit in an np.uint8    
    assert(n < 256) 
    coding = {j:i for i, j in enumerate(distinct)}
    counts = np.zeros((n, n))
    for seq in sequences:
        coded_seq = np.fromiter((coding[i] for i in seq), dtype=np.uint8)
        pairs = coded_seq[:-1] + n * coded_seq[1:]
        counts += np.bincount(pairs, minlength=n*n).reshape(n, n)
    totals = counts.sum(axis=0)
    totals[totals == 0] = 1     # avoid division by zero
    probs = counts / totals
    return {(a, b): p for a, b in product(distinct, repeat=2) 
            for p in (probs[coding[b], coding[a]],) if p}


def count_trans_mat(seq, shape):
    """
    Generate transition matrix counts. 
    """
    counts_matrix = np.zeros(shape)
    flat_coords = np.ravel_multi_index((seq[:-1], seq[1:]),
                                       counts_matrix.shape)
    return np.bincount(flat_coords,
                       minlength=counts_matrix.size).reshape(counts_matrix.shape)



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
    
