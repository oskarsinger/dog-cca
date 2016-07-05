from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu
from ... import utils as eu

import h5py

def run_online_n_view_ccalin_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    max_iter=10000,
    eta=0.1,
    exps=None, windows=None,
    num_coords=[None]*6,
    gep_max_iter=10,
    verbose=False):

    bs = cca_k + icl(cca_k)
    print "Creating data loaders"
    print "Creating data servers"
    dls = eu.get_e4_loaders(hdf5_path, subject, seconds, True)
    dss = [M2M(dl, bs, num_coords=nc) 
           for (dl, nc) in zip(dls, num_coords)]

    print "Training model"
    return ecu.run_online_n_view_ccalin_experiment(
        dss, cca_k,
        max_iter=max_iter,
        eta=eta, 
        exps=exps, windows=windows,
        gep_max_iter=gep_max_iter,
        verbose=verbose)

