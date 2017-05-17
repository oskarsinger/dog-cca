from data.servers.batch import BatchServer as BS
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu

import data.loaders.e4.shortcuts as dles

import h5py

def run_batch_n_view_ccalin_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    gep_max_iter=100,
    eta=0.1,
    exps=None, windows=None,
    percentiles=None,
    num_coords=None,
    subroutine_max_iter=1000,
    verbose=False):

    if num_coords is None:
        num_coords = [None] * 4

    print "Creating data loaders"
    dls = dles.get_e4_loaders(hdf5_path, subject, seconds, False)
    print "Creating data servers"
    dss = [BS(dl, num_coords=nc) 
           for (dl, nc) in zip(dls, num_coords)]

    print "Training model"
    return ecu.run_batch_n_view_ccalin_experiment(
        dss, cca_k,
        gep_max_iter=gep_max_iter,
        eta=eta, 
        percentiles=percentiles,
        subroutine_max_iter=subroutine_max_iter,
        verbose=verbose)

