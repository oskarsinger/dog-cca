import click
import os

import numpy as np

from whitehorses.loaders.multiview import get_easy_DCCAPMLs
from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import unzip
from theline.utils import get_quadratic, get_rotation

@click.command()
@click.option('--data-dir')
@click.option('--num-data', default=1000)
@click.option('--k', default=2)
@click.option('--ds', default=' '.join([str(10 * i) for i in range(1, 100)]))
@click.option('--pi-factors', default=' '.join(['1.5'] * 100))
@click.option('--lazy', default=True)
def run_things_all_day_bb(
    data_dir,
    num_data,
    k,
    ds,
    pi_factors,
    lazy):

    ds = [int(d) for d in ds.split()]
    angles = [float(pf) * np.pi 
              for pf in pi_factors.split()]
    pre_As = [np.random.randn(2*d, d)
              for d in ds]
    As = [np.dot(pre_A.T, pre_A)
          for pre_A in pre_As]
    (_, Qs) = unzip(
        [np.linalg.eig(A) for A in As])
    zipped = zip(
        ds, angles, Qs)
    dynamics = [get_rotation(d, a, Q, P_inv=Q.T)
                for (d, a, Q) in zipped]
    loaders = get_easy_DCCAPMLs(
        dynamics,
        num_data, 
        k,
        ds, 
        lazy=lazy) 
    names = ['N', 'k', 'ds', 'pi_factors']
    vals = [
        str(num_data), 
        str(k),
        '-'.join([str(d) for d in ds]),
        '-'.join([str(pf) for pf in pi_factors])]
    dirname = get_ts('easy_cca' + '_' + '_'.join(
        [n + '-' + v for (n, v) in zip(names, vals)]))
    dirpath = os.path.join(data_dir, dirname)

    os.mkdir(dirpath)
    
    for (i, l) in enumerate(loaders):
        filename = 'view_' + str(i) + '.csv'
        filepath = os.path.join(dirpath, filename)

        np.savetxt(filepath, l.get_data(), delimiter=',')

if __name__=='__main__':
    run_things_all_day_bb()
