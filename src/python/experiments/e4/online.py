from appgrad import AppGradCCA as AGCCA
from appgrad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders import readers
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

def run_online_appgrad_e4_data_experiment(
    dir_path, file1, file2, cca_k,
    seconds=10,
    reader1=readers.get_scalar, 
    reader2=readers.get_scalar,
    exp=False, lower=None, verbose=False):

    bs = cca_k + icl(cca_k)
    dl1 = None 
    
    if 'IBI' in file1:
        dl1 = IBI(
            dir_path, file1, seconds, reader1, online=True)
    else:
        dl1 = FRL(
            dir_path, file1, seconds, reader1, online=True)

    dl2 = None 
    
    if 'IBI' in file1:
        dl2 = IBI(
            dir_path, file2, seconds, reader2, online=True)
    else:
        dl2 = FRL(
            dir_path, file2, seconds, reader2, online=True)

    ds1 = M2M(dl1, bs)
    ds2 = M2M(dl2, bs)

    return run_online_appgrad_experiment(
        ds1, ds2, cca_k,
        exp=exp, lower=lower, verbose=verbose)

def run_n_view_online_appgrad_e4_data_experiment(
    dir_path, cca_k,
    seconds=10, exp=False, lower=None, verbose=False):

    bs = cca_k + icl(cca_k)
    mag = readers.get_magnitude
    vec = readers.get_vector
    sca = readers.get_scalar 
    dls = [
        FRL(dir_path, 'ACC.csv', seconds, mag, 32.0, online=True),
        IBI(dir_path, 'IBI.csv', seconds, vec, online=True),
        FRL(dir_path, 'BVP.csv', seconds, sca, 64.0, online=True),
        FRL(dir_path, 'TEMP.csv', seconds, sca, 4.0, online=True),
        FRL(dir_path, 'HR.csv', seconds, sca, 1.0, online=True),
        FRL(dir_path, 'EDA.csv', seconds, sca, 4.0, online=True)]
    dss = [M2M(dl, bs) for dl in dls]

    return run_online_n_view_appgrad_experiment(
        dss, cca_k,
        exp=exp, lower=lower, verbose=verbose)
