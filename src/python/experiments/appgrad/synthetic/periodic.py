import numpy as np

import data.loaders.synthetic.shortcuts as dlss
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_experiment(
    n, ps, periods, amplitudes, phases, indexes,
    max_iter=1,
    exps=None, windows=None,
    etas=None, lowers=None,
    cs=None,
    period_noise=True,
    phase_noise=True,
    amplitude_noise=True,
    verbose=False):

    bs = 3
    loaders = dlss.get_cosine_loaders(
        ps, periods, amplitudes, phases, indexes,
        period_noise, phase_noise, amplitude_noise)
    servers = [M2M(loader, bs, center=True)
               for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, 1,
        max_iter=max_iter,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        periods=periods, cs=cs,
        verbose=verbose)
