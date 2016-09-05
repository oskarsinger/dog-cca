import numpy as np
import data.loaders.e4.shortcuts as dles

from runners.bandit import FiniteHyperBandRunner as FHBR
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau

def run_n_view_online_appgrad_e4_data_hyperband_experiment(
    hdf5_path, k, subject,
    seconds=1,
    max_rounds=10):

    max_size = int(5 * 24 * 3600 / seconds)
    min_size = int(24 * 3600 / seconds)
    bs = k + icl(k) + 1

    print 'Initializing data loaders'

    dl_list = dles.get_changing_e4_loaders(
        hdf5_path, subject, seconds, True)

    print 'Initializing data servers'

    ds_list = [M2M(dl, bs, center=True) for dl in dl_list]
    dimensions = [ds.cols() for ds in ds_list]

    print 'Initializing HyperBandRunner'

    runner = FHBR(
        eau.RandomArmSampler(dimensions, k, bs).get_arm,
        ds_list,
        max_rounds,
        max_size,
        min_size)

    print 'Running Hyperband runner'

    runner.run()

    return runner
