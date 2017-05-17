from appgrad import AppGradCCA as AGCCA
from appgrad import NViewAppGradCCA as NVAGCCA
from data.loaders.e4 import FixedRateLoader as FRL
from data.loaders.e4 import IBILoader as IBI
from data.loaders.readers import from_num as fn
from data.servers.batch import BatchServer as BS

import data.loaders.e4.shortcuts as dles

import numpy as np

def test_batch_appgrad(
    ds1, ds2, cca_k):

    model = AGCCA(cca_k)

    model.fit(
        ds1, ds2,
        verbose=True)

    return model

def test_batch_n_view_appgrad(
    ds_list, cca_k):

    model = NVAGCCA(cca_k, len(ds_list))

    model.fit(
        ds_list,
        verbose=True)

    return model

def test_two_fixed_rate_scalar(
    hdf5_path, cca_k, subject, 
    sensor1, sensor2,
    seconds=1,
    reg1=0.1, reg2=0.1,
    reader1=fn.get_fields_as_columns, 
    reader2=fn.get_fields_as_columns):

    dl1 = FRL(hdf5_path, subject, sensor1, seconds, reader1)
    dl2 = FRL(hdf5_path, subject, sensor2, seconds, reader2)
    ds1 = BS(dl1)
    ds2 = BS(dl2)

    return test_batch_appgrad(
        ds1, ds2, cca_k)

def test_n_fixed_rate_scalar(
    hdf5_path, subject, cca_k,
    seconds=10):

    dls = dles.get_e4_loaders(hdf5_path, subject, seconds, False)
    dss = [BS(dl) for dl in dls]

    return test_batch_n_view_appgrad(
        dss, cca_k)
