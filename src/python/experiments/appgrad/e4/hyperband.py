import numpy as np
import numpy.random as npr
import data.loaders.e4.shortcuts as dles

from runners.bandit import FiniteHyperBandRunner as FHBR
from arms.appgrad import NView

from runners.bandit import FiniteHyperBandRunner as FHBR
from appgrad import NViewAppGradCCA as NVAGCCA
from arms.appgrad import NViewAppGradCCAArm as NVAGCCAA
from data.servers.minibatch import Minibatch2Minibatch as M2M
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BCGS
from drrobert.arithmetic import int_ceil_log as icl
from drrobert.random import log_uniform as lu
from optimization.optimizers.quasinewton import FullAdamOptimizer as FADO
from optimization.stepsize import InverseSquareRootScheduler as ISRS

def run_n_view_online_appgrad_e4_data_hyperband_experiment(
    hdf5_path, subject, k,
    max_rounds,
    max_size,
    min_size):

    return 'poop'
