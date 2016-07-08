from experiments.ccalin.e4.online import run_n_view_online_ccalin_shifting_mean_gaussian_data_experiment as rnvocsmgde
from experiments.plotting.bases import plot_canonical_bases as pcb
from experiments.plotting.filtering import plot_grouped_by_view as pgbv
from experiments.plotting.filtering import plot_grouped_by_component as pgbc
from drrobert.misc import unzip, get_nums_as_strings as n2s
from drrobert.random import normal

from random import choice

import numpy as np
import numpy.random as npr

import os

def online_randomize_or_die_son(ps, means, rates, trials=50, verbose=False):

    num_views = len(ps)

    for i in range(trials):
        # Prepare parameter choices
        cca_k = npr.randint(1,min(ps))
        (exps, windows) = (None, None)

        if choice([True, False]):
            exps = npr.uniform(
                low=0.5,high=0.99,size=num_views).tolist()
        else:
            windows = npr.randint(0,51,num_views).tolist()

        percentiles = None

        if choice([True, False]):
            upper = min(min(ps)+1, 20)
            num_coords = npr.randint(cca_k,upper,num_views).tolist()
            percentiles = [np.linspace(0,1,num=nc) for nc in num_coords]

        eta = normal(scale=0.01)[0]

        # Run experiment
        try:
            print 'Creating plot path'
            plot_path_base = '/home/oskar/GitRepos/online-cca/plots/ccalin/online/random'
            new_dir = '_'.join([
                'k',
                str(cca_k),
                'exps',
                _get_repr(exps),
                'windows',
                _get_repr(windows),
                'percentiles',
                _get_repr(num_coords),
                'eta',
                str(eta)])
            plot_path = os.path.join(plot_path_base, new_dir)
            print plot_path

            print 'Training model'
            model = run_onvcede(
                ps, cca_k, means, rates,
                exps=exps, windows=windows,
                percentiles=percentiles,
                eta=eta,
                verbose=verbose)

            print 'Creating plot directory'
            os.mkdir(plot_path)

            print 'Generating plots'
            pgbv(model, plot_path=plot_path)
            pgbv(model, plot_path=plot_path, historical=True)
            pgbc(model, plot_path=plot_path)
            pgbc(model, plot_path=plot_path, historical=True)
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
