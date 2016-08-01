from .. import utils as emu
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

import data.loaders.e4.shortcuts as dles

def run_n_view_online_periodic_appgrad_e4_data_experiment(
    hdf5_path, subject, k, seconds, num_periods,
    exp=None, window=None, 
    max_iter=10,
    verbose=False):

    bs = k + icl(k)
    period_length = (24 * 3600) / (num_periods * seconds)
    dl_list = dles.get_e4_loaders(
        hdf5_path, 
        subject, 
        seconds, 
        True)
    ds_list = [M2M(dl, bs, center=True) for dl in dl_list]
    get_learner = emu.get_appgrad_factory()
    get_gs = None
    
    if (exp is not None) and (window is not None):
        raise ValueError(
            'Parameters exp and window cannot both be set.')
    elif exp is not None:
        get_gs = emu.get_exp_gs_factory(exp)
    elif window is not None:
        get_gs = emu.get_boxcar_gs_factory(window)

    return emu.run_n_view_periodic_metalearner_experiment(
            ds_list, k, get_learner, period_length, num_periods,
            max_iter=max_iter,
            get_gs=get_gs, verbose=verbose)
