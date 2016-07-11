from .. import utils as emu
from ... import utils as eu
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

def run_n_view_online_periodic_appgrad_e4_data_experiment(
    hdf5_path, subject, k, seconds, num_periods,
    exp=None, window=None, 
    verbose=False):

    bs = k + icl(k)
    period_length = (24 * 3600) / num_periods
    dl_list = eu.get_hr_and_acc(
        hdf5_path, 
        subject, 
        seconds, 
        True)
    ds_list = [M2M(dl, batch_size=bs) for dl in dl_list]
    get_learner = emu.get_appgrad_factory()
    get_gs = None
    
    if (exp is not None) and (window is not None):
        raise ValueError(
            'Parameters exp and window cannot both be set.')
    elif exp is not None:
        get_gs = emu.get_exp_gs_factory(exp)
    elif window is not None:
        get_gs = emu.get_boxcar_gs_factory(window)

    return emu.run_online_n_view_periodic_metalearner_experiment(
            ds_list, k, get_learner, period_length, num_periods,
            get_gs=get_gs, verbose=verbose)
