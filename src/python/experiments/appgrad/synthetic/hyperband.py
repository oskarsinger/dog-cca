import numpy as np
import data.loaders.synthetic.shortcuts as dlss

from runners.bandit import FiniteHyperBandRunner as FHBR
from arms.appgrad import NViewAppGradCCAArmGenerator as NVAGCCAAG
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_hyperband_experiment(
    n, ps, periods, amplitudes, phases, indexes,
    max_rounds,
    max_size,
    min_size):

    dl_list = dlss.get_cosine_loaders(
        ps, periods, amplitudes, phases, indexes,
        period_noise, phase_noise, amplitude_noise)
    # TODO: offer different centering options
    # 1. long term physiological averages
    # 2. moving averages; new hyperparameter for forget factor; ugh
    ds_list = [M2M(loader, bs, center=True) 
               for loader in loaders]
    runner = FHBR(
        get_random_arm,
        ds_list,
        max_rounds,
        max_size,
        min_size)

    runner.run()

def get_random_arm():

    # Things to sample
    beta1 = None
    beta2 = None
    stepsize = None
    window = None
    exp = None
    

    return 'poop'
