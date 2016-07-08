from experiments.appgrad.e4.online import run_n_view_online_appgrad_shifting_mean_gaussian_data_experiment as rnvoasmgde
from experiments.plotting.bases import plot_canonical_bases as pcb
from experiments.plotting.filtering import plot_grouped_by_view as pgbv
from experiments.plotting.filtering import plot_grouped_by_component as pgbc
from drrobert.misc import unzip, get_nums_as_strings as n2s

from random import choice

import numpy as np
import numpy.random as npr

import os

def online_randomize_or_die_son(num_views, trials=50):

    for i in range(trials):
        # Prepare parameter choices
        cca_k = npr.randint(1,10)
        (exps, windows) = (None, None)

        if choice([True, False]):
            exps = npr.uniform(
                low=0.5,high=0.99,size=4).tolist()
        else:
            windows = npr.randint(0,51,4).tolist()

        upper = min(seconds+1, 20)
        num_coords = npr.randint(cca_k,upper,4).tolist()
        percentiles = [np.linspace(0,1,num=nc)
                       for nc in num_coords]
        etas = np.absolute(npr.randn(4)).tolist()

        # Run experiment
        try:
            print "Training model"
            model = rnvoasmgde(
                ps, cca_k, means, rates,
                exps=exps, windows=windows,
                percentiles=percentiles,
                etas=etas)
            print "Creating plot path"
            plot_path_base = '/home/oskar/GitRepos/online-cca/plots/appgrad/'
            new_dir = '_'.join([
                'k',
                str(cca_k),
                'seconds',
                str(seconds),
                'exp',
                'None' if exps is None else '-'.join(n2s(exps)),
                'windows',
                'None' if windows is None else '-'.join(n2s(windows)),
                'num_coords',
                '-'.join(n2s(num_coords)),
                'etas',
                '-'.join(n2s(etas))])
            plot_path = os.path.join(plot_path_base, new_dir)

            print "Generating plots"
            os.mkdir(plot_path)
            pgbv(model, plot_path=plot_path)
            pgbv(model, historical=True, plot_path=plot_path)
            pgbc(model, plot_path=plot_path)
            pgbc(model, historical=True, plot_path=plot_path)
            pcb(model, plot_path=plot_path)
        except Exception as e:
            print e 
