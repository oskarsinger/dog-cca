from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.readers import from_num as fn
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu

import h5py

def run_online_n_view_ccalin_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    max_iter=10000,
    eta=0.1,
    exps=None, windows=None,
    num_coords=[None]*6,
    gep_max_iter=100,
    verbose=False):

    bs = cca_k + icl(cca_k)
    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns
    print "Creating data loaders"
    dls = [
        FRL(hdf5_path, subject, 'ACC', seconds, mag, online=True),
        IBI(hdf5_path, subject, 'IBI', seconds, fac, online=True),
        FRL(hdf5_path, subject, 'BVP', seconds, fac, online=True),
        FRL(hdf5_path, subject, 'TEMP', seconds, fac, online=True),
        FRL(hdf5_path, subject, 'HR', seconds, fac, online=True),
        FRL(hdf5_path, subject, 'EDA', seconds, fac, online=True)]
    print "Creating data servers"
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

