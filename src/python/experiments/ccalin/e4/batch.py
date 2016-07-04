from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.readers import from_num as fn
from data.servers.batch import BatchServer as BS
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as ecu

import h5py

def run_n_view_ccalin_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    max_iter=10000,
    eta=0.1,
    exps=None, windows=None,
    num_coords=[None]*6,
    gep_max_iter=100,
    verbose=False):

    mag = fn.get_row_magnitude
    fac = fn.get_fields_as_columns
    print "Creating data loaders"
    dls = [
        FRL(hdf5_path, subject, 'ACC', seconds, mag),
        IBI(hdf5_path, subject, 'IBI', seconds, fac),
        FRL(hdf5_path, subject, 'BVP', seconds, fac),
        FRL(hdf5_path, subject, 'TEMP', seconds, fac),
        FRL(hdf5_path, subject, 'HR', seconds, fac),
        FRL(hdf5_path, subject, 'EDA', seconds, fac)]
    print "Creating data servers"
    dss = [BS(dl, num_coords=nc) 
           for (dl, nc) in zip(dls, num_coords)]

    print "Training model"
    return ecu.run_batch_n_view_ccalin_experiment(
        dss, cca_k,
        max_iter=max_iter,
        eta=eta, 
        gep_max_iter=gep_max_iter,
        verbose=verbose)

