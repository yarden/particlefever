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

def run_model(data):
    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            alpha=6.,gamma=6.,
            init_state_concentration=6.,
            obs_distns=obs_distns,
            dur_distns=dur_distns)
    posteriormodel.add_data(small_data)
    models = []
    for idx in progprint_xrange(200):
        posteriormodel.resample_model()
        if (idx+1) % 10 == 0:
            models.append(copy.deepcopy(posteriormodel))
    return models

# strongly periodic data, except for first four observations
small_data = np.array([0, 0]*2 + [0, 1]*10)
big_data = np.array([0, 0]*2 + [0, 1]*50)

# inference
small_models = run_model(small_data)
big_models = run_model(big_data)

def summarize_pred(models, num_preds=10):
    all_preds = []
    max_state = 1
    for m in models:
        prediction, hidden_states = m.predict(np.array([]), num_preds)
        max_state = max(max_state, hidden_states.max())
        print "prediction: ", prediction
        print "prediction: ", len(prediction)
        all_preds.append(prediction)
    all_preds = np.array(all_preds)
    mean_preds = all_preds.mean(axis=0)
    print "mean preds: ", mean_preds, len(mean_preds)
    return mean_preds, max_state

print "prediction posterior for small data: "
small_summary, small_max_state = summarize_pred(small_models)
print small_summary
print " - max # states: %d" %(small_max_state)
print "prediction posterior for big data: "
big_summary, big_max_state = summarize_pred(big_models)
print big_summary
print " - max # states: %d" %(big_max_state)

### plotting
# import seaborn as sns
# plt.figure()
# sns.set_style("ticks")
# x_axis = range(len(small_summary))
# plt.plot(x_axis, small_summary, label="small")
# plt.xticks(x_axis)
# plt.ylim(0, 1)
# plt.plot(x_axis, big_summary, label="big")
# plt.xticks(x_axis)
# plt.ylim(0, 1)
# plt.legend(loc="upper right")
# plt.ylabel("P(next output | data)")
# sns.despine(trim=True, offset=2)
# plt.show()


# occassional error:
#     323     def resample(self):
#     324         betal, betastarl = self.messages_backwards()
# --> 325         self.sample_forwards(betal,betastarl)
#     326 
#     327     def copy_sample(self,newmodel):

# /Users/yarden/my_projects/pyhsmm/pyhsmm/internals/hsmm_states.pyc in sample_forwards(self, betal, betastarl)
#     526                 self.trans_matrix,caBl,self.aDl,self.pi_0,betal,betastarl,
#     527                 np.empty(betal.shape[0],dtype='int32'))
# --> 528         assert not (0 == self.stateseq).all()
#     529 
#     530     def sample_forwards_python(self,betal,betastarl):

