from data.loaders.random import GaussianLoader as GL
from data.servers.minibatch import Batch2Minibatch as B2M
from drrobert.arithmetic import int_ceil_log as icl

import experiments.utils as aeu

import numpy as np

def run_online_appgrad_random_data_experiment(
    p1, p2, k, exp=False, lower=None):

    print "Parameters:\n\t", "\n\t".join([
        "batch_size: " + str(batch_size),
        "p1: " + str(p1),
        "p2: " + str(p2),
        "k: " + str(k)])

    bs = k + icl(k)
    X_loader = GL(10*p1, p1)
    Y_loader = GL(10*p2, p2)
    X_server = B2M(X_loader, bs)
    Y_server = B2M(Y_loader, bs)

    print "Testing CCA with boxcar-weighted Gram matrices"
    return aeu.run_online_appgrad_experiment(
        X_server, Y_server, k, exp=exp, lower=lower)

def run_n_view_online_appgrad_random_data_experiment(
    ps, k, exp=False, lower=None):

    print "Gaussian random data online AppGrad CCA tests"
    print "Parameters:\n\t", "\n\t".join([
        "ps: " + str(ps),
        "k: " + str(k)])

    bs = k + icl(k)
    loaders = [GL(10*p, p) for p in ps]
    servers = [B2M(loader, bs) for loader in loaders]

    print "Testing n-view CCA with boxcar-weighted Gram matrices"
    return aeu.run_online_n_view_appgrad_experiment(
        servers, k, exp=exp, lower=lower)
