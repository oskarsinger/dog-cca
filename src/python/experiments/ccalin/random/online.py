import numpy as np

from data.loaders.random import GaussianLoader as GL
from data.servers.minibatch import Batch2Minibatch as B2M
from drrobert.arithmetic import int_ceil_log as icl

import experiments.ccalin.utils as ecu

def run_n_view_online_ccalin_random_data_experiment(
    ps, k,
    max_iter=10000,
    eta=0.1,
    exp=False, verbose=False,
    means=None, eta=0.1):

    bs = k + icl(k)
    loaders = [GL(10*p, p, means=mean)
               for (mean, p) in zip(means, ps)]
    servers = [B2M(loader, bs) for loader in loaders]

    return ecu.run_online_n_view_ccalin_experiment(
        servers, k,
        max_iter=max_iter,
        eta=eta,
        exp=exp, 
        verbose=verbose)
