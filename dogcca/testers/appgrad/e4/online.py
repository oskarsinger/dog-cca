from appgrad import AppGradCCA as AGCCA
from appgrad import NViewAppGradCCA as NVAGCCA

from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau

import data.loaders.e4.shortcuts as dles 

import h5py

def run_n_view_online_appgrad_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    exps=None, windows=None,
    cs=None, periods=None,
    keep_basis_history=False,
    verbose=False,
    etas=None, lowers=None):

    bs = cca_k + icl(cca_k)
    print "Creating data loaders"
    dls = dles.get_hr_and_acc(
        hdf5_path, subject, seconds, True)
    print "Creating data servers"
    dss = [M2M(dl, bs, center=True) for dl in dls]

    print "Training model"
    return eau.run_online_n_view_appgrad_experiment(
        dss, cca_k,
        exps=exps, windows=windows,
        cs=cs, periods=periods,
        lowers=lowers, etas=etas, 
        keep_basis_history=keep_basis_history,
        verbose=verbose)

def run_all_subject_n_view_online_appgrad_e4_data_experiment(
    hdf5_path, cca_k,
    seconds=10,
    exp=False, verbose=False,
    etas=None, lowers=None):

    subjects = h5py.File(hdf5_path).keys()
    models = []

    for subject in subjects:
        print 'Training model for subject', subject + '.'

        model = None

        try:
            model = run_n_view_online_appgrad_e4_data_experiment(
                hdf5_path, cca_k, subject, 
                seconds=seconds,
                exp=exp, 
                verbose=verbose, 
                etas=etas, 
                lowers=lowers)
        except KeyError:
            print 'Subject', subject, 'was missing a view.'

        models.append(model)

    return models
