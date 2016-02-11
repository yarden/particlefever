##
## Test pyhsmm
##
import os
import sys
import time
import copy

import numpy as np

import matplotlib.pylab as plt

import pyhsmm
import pyhsmm.basic.distributions as distributions
import pyhsmm.util.text
from pyhsmm.util.text import progprint_xrange 

Nmax = 25

# observations
obs_hypparams = {"K": 2,
                 "alpha_0": 1.}
# gamma prior on duration 
dur_hypparams = {'alpha_0': 1.,
                 'beta_0': 1.}
obs_distns = [distributions.Categorical(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6.,gamma=6.,
        init_state_concentration=6.,
        obs_distns=obs_distns,
        dur_distns=dur_distns)

# data
#data = np.array([0, 1] * 10 + [0, 0] * 10)
data = np.array([0, 0] * 10 + [0, 1] * 10)
posteriormodel.add_data(data)

# inference
models = []
for idx in progprint_xrange(200):
    posteriormodel.resample_model()
    if (idx+1) % 10 == 0:
        models.append(copy.deepcopy(posteriormodel))

# try to predict (fails)
for m in models:
    print m.predict(data, 10)[0]

def summarize_pred(models, num_preds=10):
    all_preds = []
    for m in models:
        all_preds.append(m.predict(np.array([]), num_preds)[0])
    all_preds = np.array(all_preds)
    mean_preds = all_preds.mean(axis=1)
    return mean_preds

print "mean preds: "
print summarize_pred(models)

