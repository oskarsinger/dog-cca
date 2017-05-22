from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu

import data.loaders.e4.shortcuts as dles

import h5py

def run_online_n_view_ccalin_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    max_iter=10000,
    eta=0.1,
    exps=None, windows=None,
    percentiles=None,
    num_coords=None,
    gep_max_iter=10,
    verbose=False):

    if num_coords is None:
        num_coords=[None] * 4

    bs = cca_k + icl(cca_k)
    print "Creating data loaders"
    dls = dles.get_e4_loaders(hdf5_path, subject, seconds, True)
    print "Creating data servers"
    dss = [M2M(dl, bs, center=True, num_coords=nc) 
           for (dl, nc) in zip(dls, num_coords)]

    print "Training model"
    return ecu.run_online_n_view_ccalin_experiment(
        dss, cca_k,
        max_iter=max_iter,
        eta=eta, 
        exps=exps, windows=windows,
        percentiles=percentiles,
        gep_max_iter=gep_max_iter,
        verbose=verbose)
