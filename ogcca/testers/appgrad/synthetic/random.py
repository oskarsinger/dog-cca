import numpy as np

from data.loaders.synthetic import ShiftingMeanGaussianLoader as SMGL
from data.loaders.synthetic import GaussianLoader as GL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from data.servers.minibatch import Batch2Minibatch as B2M
from drrobert.arithmetic import int_ceil_log as icl

from .. import utils as eau

def run_n_view_online_appgrad_gaussian_data_experiment(
    n, ps, cca_k,
    max_iter=10,
    percentiles=None,
    exps=None, windows=None,
    etas=None, lowers=None,
    verbose=False):

    bs = cca_k + icl(cca_k)
    loaders = [GL(n, p) for p in ps]
    servers = [B2M(loader, bs, center=True) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, cca_k,
        max_iter=max_iter,
        percentiles=percentiles,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        verbose=verbose)

def run_n_view_online_appgrad_shifting_mean_gaussian_data_experiment(
    ps, cca_k, means, rates,
    max_iter=10,
    percentiles=None,
    exps=None, windows=None,
    etas=None, lowers=None,
    verbose=False):

    bs = cca_k + icl(cca_k)
    loaders = [SMGL(p, mean, rate)
               for (p, mean, rate) in zip(ps, means, rates)]
    servers = [M2M(loader, bs, center=True) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, cca_k,
        max_iter=max_iter,
        percentiles=percentiles,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        verbose=verbose)
