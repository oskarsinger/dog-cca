from appgrad import AppGradCCA as AGCCA
from appgrad import NViewAppGradCCA as NVAGCCA

from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau
from ... import utils as eu

import h5py

def run_n_view_online_appgrad_e4_data_experiment(
    hdf5_path, cca_k, subject,
    seconds=10, 
    exps=None, windows=None,
    num_coords=[None]*6,
    verbose=False,
    etas=None, lowers=None):

    bs = cca_k + icl(cca_k)
    print "Creating data loaders"
    dls = eu.get_e4_loaders(hdf5_path, subject, seconds, True)
    print "Creating data servers"
    dss = [M2M(dl, bs, num_coords=nc)
           for (dl, nc) in zip(dls, num_coords)]

    print "Training model"
    return eau.run_online_n_view_appgrad_experiment(
        dss, cca_k,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas, 
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
