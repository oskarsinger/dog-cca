import numpy as np
import numpy.random as npr
import data.loaders.synthetic.shortcuts as dlss

from runners.bandit import FiniteHyperBandRunner as FHBR
from appgrad import NViewAppGradCCA as NVAGCCA
from arms.nappgrad import NViewAppGradCCAArm as NVAGCCAA
from data.servers.minibatch import Minibatch2Minibatch as M2M
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BCGS
from drrobert.arithmetic import int_ceil_log as icl
from drrobert.random import log_uniform as lu
from optimization.optimizers.quasinewton import FullAdamOptimizer as FADO
from optimization.stepsize import InverseSquareRootScheduler as ISRS
from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_hyperband_experiment(
    k, n, ps, 
    periods, amplitudes, phases, indexes,
    max_rounds=10):

    max_size = n
    min_size = int(min(periods)) * 3
    bs = cca_k + icl(cca_k) + 1
    dl_list = dlss.get_cosine_loaders(
        ps, periods, amplitudes, phases, indexes,
        period_noise, phase_noise, amplitude_noise)
    ds_list = [M2M(loader, bs, center=True) 
               for loader in loaders]
    dimensions = [ds.cols() for ds in ds_list]
    runner = FHBR(
        eau.RandomArmSampler(dimensions, k, bs).get_arm,
        ds_list,
        max_rounds,
        max_size,
        min_size)

    runner.run()

    return runner
