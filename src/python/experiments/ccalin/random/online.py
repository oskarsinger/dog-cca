import numpy as np

from data.loaders.random import GaussianLoader as GL
from data.servers.minibatch import Batch2Minibatch as B2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu

def run_n_view_online_ccalin_random_data_experiment(
    ps, k,
    max_iter=10000,
    eta=0.1,
    exps=None, verbose=False,
    means_list=None):

    if means_list is None:
        means_list = [None] * len(ps)

    bs = k + icl(k)
    loaders = [GL(10*p, p, means=means)
               for (means, p) in zip(means_list, ps)]
    servers = [B2M(loader, bs) for loader in loaders]

    return ecu.run_online_n_view_ccalin_experiment(
        servers, k,
        max_iter=max_iter,
        eta=eta,
        exps=exps, 
        verbose=verbose)
