##
## Test simple HMM model
##
import os
import sys
import time

import pyhsmm
import pyhsmm.models as pyhsmm_models

import particlefever
import particlefever.dbn as dbn
import particlefever.models as models

def make_rain_hmm():
    """
    Make the umbrella HMM from AIMA.
    """
    # Initial model containing the rain node (hidden)
    # and the umbrella node (observation)
    import models.rain_hmm_model as rain_hmm_model


def hmm_with_pyhsmm():
    """
    Run inference using pyhsmm on a simple HMM.
    """
    pass


def hmm_with_partfever():
    """
    Run inference using particle fever.
    """
    pass



def main():
    make_rain_hmm()



if __name__ == "__main__":
    main()
