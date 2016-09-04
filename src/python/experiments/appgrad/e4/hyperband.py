import numpy as np
import numpy.random as npr
import data.loaders.e4.shortcuts as dles

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

def run_n_view_online_appgrad_e4_data_hyperband_experiment(
    hdf5_path, cca_k, subject,
    seconds=1,
    max_rounds=10):

    max_size = int(5 * 24 * 3600 / seconds)
    min_size = int(24 * 3600 / seconds)
    bs = cca_k + icl(cca_k) + 1
    dl_list = dles.get_changing_e4_loaders(
        hdf5_path, subject, seconds, True)
    ds_list = [M2M(dl, bs, center=True) for dl in dls]
    dimensions = [ds.cols() for ds in ds_list]
    runner = FHBR(
        eau.RandomArmSampler(dimensions, k, bs).get_arm,
        ds_list,
        max_rounds,
        max_size,
        min_size)

    runner.run()

    return runner
