##
## Statistics utilities
##
import scipy
import scipy.special
from scipy.special import gammaln

def C(alpha):
    """
    Following notation from:
    http://users.cecs.anu.edu.au/~ssanner/MLSS2010/Johnson1.pdf
    
    C(alpha) = [\prod_j=1^{m}\gamma(\alpha_j)] /
               \gamma(\sum_{j=1}^{m}\alpha_j)

    logC(alpha) = log(\prod_j=1^{m}\gamma(\alpha_j)]) - \
                  log(\gamma(\sum_{j=1}^{m}\alpha_j))
                  
                = \sum_{j=1}^{m}\gammaln(\alpha_j) - \
                  \gammaln(\sum_{j=1}^{m}\alpha_j))
    """
    # compute C function in log form:
    logC = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    return logC
      

