import numpy as np

from data.loaders.random import GaussianLoader as GL
from data.servers.minibatch import Batch2Minibatch as B2M
from drrobert.arithmetic import int_ceil_log as icl

import experiments.appgrad.utils as eau

def run_online_appgrad_random_data_experiment(
    p1, p2, k, 
    exp=False, verbose=False, whiten=False,
    lower1=None, lower2=None, means=None, etas=None):

    bs = k + icl(k)
    X_loader = GL(10*p1, p1, means=means)
    Y_loader = GL(10*p2, p2, means=means)
    X_server = B2M(X_loader, bs, whiten=whiten)
    Y_server = B2M(Y_loader, bs, whiten=whiten)

    return eau.run_online_appgrad_experiment(
        X_server, Y_server, k,
        exp=exp, 
        lower1=lower1, lower2=lower2,
        verbose=verbose, etas=etas)

def run_n_view_online_appgrad_random_data_experiment(
    ps, k, 
    exp=False, verbose=False, whiten=False,
    lowers=None, means=None, etas=None):

    bs = k + icl(k)
    loaders = [GL(10*p, p, means=mean)
               for mean, p in zip(means, ps)]
    servers = [B2M(loader, bs, whiten=whiten) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, k,
        exp=exp, lowers=lowers, verbose=verbose, etas=etas)
