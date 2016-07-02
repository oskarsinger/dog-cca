from experiments.ccalin.e4.online import run_n_view_online_ccalin_e4_data_experiment as run_nvocede
from experiments.plotting.bases import plot_canonical_bases as pcb
from experiments.plotting.filtering import plot_grouped_by_view as pgbv
from experiments.plotting.filtering import plot_grouped_by_component as pgbc
from drrobert.misc import unzip, get_nums_as_strings as n2s

from random import choice

import numpy as np
import numpy.random as npr

import os

def ccalin_randomize_or_die_son(hdf5_path, subject, trials=50):

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
        num_coords = npr.randint(1,upper,6).tolist()
        eta = np.absolute(npr.randn(1))[0]

        # Run experiment
        try:
            print "Training model"
            model = run_nvoaede(
                hdf5_path, cca_k, subject,
                seconds=seconds, 
                exps=exps, windows=windows,
                num_coords=num_coords,
                eta=eta)
            print "Creating plot path"
            plot_path_base = '/home/oskar/GitRepos/online-cca/plots/ccalin/'
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
                'eta',
                eta])
            plot_path = os.path.join(plot_path_base, new_dir)

            print "Generating plots"
            os.mkdir(plot_path)
            pgbv(model, plot_path=plot_path)
            pgbc(model, plot_path=plot_path)
            pcb(model, plot_path=plot_path)
        except Exception as e:
            print e 
