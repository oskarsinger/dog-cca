import numpy as np

from data.loaders.synthetic import ShiftingMeanGaussianLoader as SMGL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu

def run_n_view_online_ccalin_shifting_mean_gaussian_data_experiment(
    ps, cca_k, means, rates,
    max_iter=10000,
    eta=0.1,
    exps=None, windows=None,
    percentiles=None,
    gep_max_iter=10,
    verbose=False):

    bs = cca_k + icl(cca_k)
    loaders = [SMGL(p, mean, rate)
               for (p, mean, rate) in zip(ps, means, rates)]
    servers = [M2M(loader, bs, center=True) for loader in loaders]

    return ecu.run_online_n_view_ccalin_experiment(
        servers, cca_k,
        max_iter=max_iter,
        eta=eta,
        exps=exps, windows=windows,
        percentiles=percentiles,
        gep_max_iter=gep_max_iter,
        verbose=verbose)
