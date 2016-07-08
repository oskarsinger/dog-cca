from experiments.ccalin.e4.online import run_online_n_view_ccalin_e4_data_experiment as run_onvcede
from experiments.plotting.bases import plot_canonical_bases as pcb
from experiments.plotting.filtering import plot_grouped_by_view as pgbv
from experiments.plotting.filtering import plot_grouped_by_component as pgbc
from drrobert.misc import unzip, get_nums_as_strings as n2s
from drrobert.random import normal

from random import choice

import numpy as np
import numpy.random as npr

import os

def online_ccalin_randomize_or_die_son(hdf5_path, subject, gep_max_iter=100, trials=50, verbose=False):

    for i in range(trials):
        # Prepare parameter choices
        cca_k = npr.randint(1,10)
        seconds = npr.randint(cca_k,51)
        (exps, windows) = (None, None)

        if choice([True, False]):
            exps = npr.uniform(
                low=0.5,high=0.99,size=6).tolist()
        else:
            windows = npr.randint(0,51,6).tolist()

        upper = min(seconds+1, 20)
        num_coords = npr.randint(cca_k,upper,6).tolist()
        percentiles = None

        #if choice([True, False]):
        percentiles = [np.linspace(0,1,num=nc) for nc in num_coords]
        #num_coords = None

        eta = normal(scale=0.01)[0]

        # Run experiment
        try:
            print 'Creating plot path'
            plot_path_base = '/home/oskar/GitRepos/online-cca/plots/ccalin/online/'
            new_dir = '_'.join([
                'k',
                str(cca_k),
                'seconds',
                str(seconds),
                'exps',
                _get_repr(exps),
                'windows',
                _get_repr(windows),
                #'num_coords',
                #_get_repr(num_coords),
                'percentiles',
                _get_repr(num_coords),
                'eta',
                str(eta)])
            plot_path = os.path.join(plot_path_base, new_dir)
            print plot_path

            print 'Training model'
            model = run_onvcede(
                hdf5_path, cca_k, subject,
                seconds=seconds, 
                exps=exps, windows=windows,
                percentiles=percentiles,
                num_coords=num_coords,
                eta=eta,
                gep_max_iter=gep_max_iter,
                verbose=verbose)

            print 'Creating plot directory'
            os.mkdir(plot_path)

            print 'Generating plots'
            pgbv(model, plot_path=plot_path)
            pgbc(model, plot_path=plot_path)
            pcb(model, plot_path=plot_path)
        except Exception as e:
            print e 

def batch_ccalin_randomize_or_die_son(hdf5_path, subject, trials=50, verbose=False):

    for i in range(trials):
        # Prepare parameter choices
        cca_k = npr.randint(1,10)
        seconds = npr.randint(cca_k,51)
        upper = min(seconds+1, 20)
        num_coords = npr.randint(cca_k,upper,6).tolist()
        percentiles = None

        #if choice([True, False]):
        percentiles = [np.linspace(0,1,num=nc).tolist() 
                       for nc in num_coords]
        #num_coords = None

        eta = normal(scale=0.01)[0]

        # Run experiment
        try:
            print 'Creating plot path'
            plot_path_base = '/home/oskar/GitRepos/online-cca/plots/ccalin/batch/'
            new_dir = '_'.join([
                'k',
                str(cca_k),
                'seconds',
                str(seconds),
                #'num_coords',
                #_get_repr(num_coords),
                'percentiles',
                _get_repr(num_coords),
                'eta',
                str(eta)])
            plot_path = os.path.join(plot_path_base, new_dir)
            print plot_path

            print 'Training model'
            model = run_bnvcede(
                hdf5_path, cca_k, subject,
                seconds=seconds, 
                num_coords=num_coords,
                percentiles=percentiles,
                eta=eta,
                verbose=verbose)

            print 'Creating plot directory'
            os.mkdir(plot_path)

            print 'Generating filter grouped by view plots'
            pgbv(model, plot_path=plot_path)
            print 'Generating filter grouped by component plots'
            pgbc(model, plot_path=plot_path)
            print 'Generating bases heat map plots'
            pcb(model, plot_path=plot_path)
        except Exception as e:
            print e 

def _get_repr(l):

    output = 'None'

    if l is not None:
        if type(l[0]) is list:
            l = ['-'.join(n2s(x)) for x in l]
            output = '_'.join(l)
        else:
            output = '-'.join(n2s(l))

    return output
