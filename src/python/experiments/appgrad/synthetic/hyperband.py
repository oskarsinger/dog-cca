import numpy as np

from runners.bandit import FiniteHyperBandRunner as FHBR
from arms.appgrad import NViewAppGradCCAArmGenerator as NVAGCCAAG
from data.loaders.synthetic import CosineLoader as CL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_hyperband_experiment(
    n, ps, periods, amplitudes, phases, indexes,
    max_rounds,
    max_size,
    min_size):



    runner = FHBR(
        get_random_arm,
        ds_list,
        max_rounds,
        max_size,
        min_size)

def get_random_arm():

    return 'poop'
