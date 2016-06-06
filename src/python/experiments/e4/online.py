from appgrad import AppGradCCA as AGCCA
from appgrad import NViewAppGradCCA as NVAGCCA

from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders import readers
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

import experiments.utils as eu

def run_online_appgrad_e4_data_experiment(
    hdf5_path, cca_k, subject, 
    sensor1, sensor2, 
    seconds=10,
    reader1=readers.get_scalar, 
    reader2=readers.get_scalar,
    exp=False, whiten=False, verbose=False,
    etas=None, lower1=None, lower2=None):

    bs = cca_k + icl(cca_k)
    dl1 = None 
    
    if 'IBI' in file1:
        dl1 = IBI(
            hdf5_path, subject, sensor1, seconds, reader1, 
            online=True)
    else:
        dl1 = FRL(
            hdf5_path, subject, sensor1, seconds, reader1, 
            online=True)

    dl2 = None 
    
    if 'IBI' in file1:
        dl2 = IBI(
            hdf5_path, subject, sensor2, seconds, reader2, 
            online=True)
    else:
        dl2 = FRL(
            hdf5_path, subject, sensor2, seconds, reader2, 
            online=True)

    ds1 = M2M(dl1, bs, whiten=whiten)
    ds2 = M2M(dl2, bs, whiten=whiten)

    return eu.run_online_appgrad_experiment(
        ds1, ds2, cca_k,
        exp=exp, 
        lower1=lower1, lower2=lower2,
        verbose=verbose, etas=etas)

def run_n_view_online_appgrad_e4_data_experiment(
    dir_path, cca_k, subject,
    seconds=10, 
    exp=False, verbose=False, whiten=False,
    etas=None, lowers=None):

    bs = cca_k + icl(cca_k)
    mag = readers.get_magnitude
    vec = readers.get_vector
    sca = readers.get_scalar 
    dls = [
        FRL(hdf5_path, subject, 'ACC', seconds, mag, online=True),
        IBI(hdf5_path, subject, 'IBI', seconds, vec, online=True),
        FRL(hdf5_path, subject, 'BVP', seconds, sca, online=True),
        FRL(hdf5_path, subject, 'TEMP', seconds, sca, online=True),
        FRL(hdf5_path, subject, 'HR', seconds, sca, online=True),
        FRL(hdf5_path, subject, 'EDA', seconds, sca, online=True)]
    dss = [M2M(dl, bs, whiten=whiten) for dl in dls]

    return eu.run_online_n_view_appgrad_experiment(
        dss, cca_k,
        exp=exp, lowers=lowers, verbose=verbose, etas=etas)
