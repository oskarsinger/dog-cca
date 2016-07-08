import numpy as np

from data.loaders.random import ShiftingMeanGaussianLoader as SMGL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

from .. import utils as eau

def run_n_view_online_appgrad_shifting_mean_gaussian_data_experiment(
    ps, cca_k, means, rates,
    percentiles=None,
    exps=None, windows=None,
    etas=None, lowers=None,
    verbose=False):

    bs = cca_k + icl(cca_k)
    loaders = [SMGL(p, mean, rate)
               for (p, mean, rate) in zip(ps, means, rates)]
    servers = [M2M(loader, bs) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, cca_k,
        percentiles=percentiles,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        verbose=verbose)
