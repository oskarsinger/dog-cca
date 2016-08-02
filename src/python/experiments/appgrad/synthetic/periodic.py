import numpy as np

from data.loaders.synthetic import CosineLoader as CL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_experiment(
    n, num_views,
    max_iter=10,
    exps=None, windows=None,
    etas=None, lowers=None,
    verbose=False):

    bs = 3
    loaders = [CL() for i in range(num_views)]
    servers = [M2M(loader, bs, center=True) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, 1,
        max_iter=max_iter,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        verbose=verbose)
