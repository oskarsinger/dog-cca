from appgrad import AppGradCCA as AGCCA
from appgrad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.readers import from_num as fn
from data.servers.batch import BatchServer as BS
from linal.utils import quadratic as quad

import numpy as np

def test_batch_appgrad(
    ds1, ds2, cca_k):

    model = AGCCA(cca_k)

    model.fit(
        ds1, ds2,
        verbose=True)

    return model.get_bases()

def test_batch_n_view_appgrad(
    ds_list, cca_k):

    model = NVAGCCA(cca_k, len(ds_list))

    model.fit(
        ds_list,
        verbose=True)

    return model.get_bases()

def test_two_fixed_rate_scalar(
    hdf5_path, cca_k, subject, 
    sensor1, sensor2,
    seconds=1,
    reg1=0.1, reg2=0.1,
    reader1=fn.get_scalar_as_is, 
    reader2=fn.get_scalar_as_is):

    dl1 = FRL(hdf5_path, subject, sensor1, seconds, reader1)
    dl2 = FRL(hdf5_path, subject, sensor2, seconds, reader2)
    ds1 = BS(dl1)
    ds2 = BS(dl2)
    (Phi, unn_Phi, Psi, unn_Psi) = test_batch_appgrad(
        ds1, ds2, cca_k)
    I_k = np.identity(cca_k)
    gram1 = ds1.get_batch_and_gram()[1]
    gram2 = ds2.get_batch_and_gram()[1]

    print np.linalg.norm(quad(Phi, gram1) - I_k)
    print np.linalg.norm(quad(Psi, gram2) - I_k)

    return (Phi, Psi)

def test_n_fixed_rate_scalar(
    hdf5_path, cca_k,
    seconds=10):

    mag = fn.get_magnitude
    vec = fn.get_vec_as_list
    sca = fn.get_scalar_as_is
    dls = [
        FRL(hdf5_path, subject, 'ACC', seconds, mag),
        IBI(hdf5_path, subject, 'IBI', seconds, vec),
        FRL(hdf5_path, subject, 'BVP', seconds, sca),
        FRL(hdf5_path, subject, 'TEMP', seconds, sca),
        FRL(hdf5_path, subject, 'HR', seconds, sca),
        FRL(hdf5_path, subject, 'EDA', seconds, sca)]
    dss = [BS(dl) for dl in dls]
    (basis_pairs, Psi) = test_batch_n_view_appgrad(
        dss, cca_k)

    return (basis_pairs, Psi)
