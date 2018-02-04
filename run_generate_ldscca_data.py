import click
import os

import numpy as np

from whitehorses.loaders.multiview.cca import get_lds_SCCAPMLs
from drrobert.file_io import get_timestamped as get_ts
from theline.utils import get_quadratic, get_rotation

@click.command()
@click.option('--data-dir')
@click.option('--num-data', default=1000)
@click.option('--k', default=2)
@click.option('--ds', default=' '.join([str(10 * i) for i in range(1, 100)]))
@click.option('--pi-factor', default=1.5)
@click.option('--seed-factor', default=5.0)
@click.option('--lazy', default=True)
def run_things_all_day_bb(
    data_dir,
    num_data,
    k,
    ds,
    pi_factor,
    seed_factor,
    lazy):

    angle = pi_factor * np.pi
    pre_A = np.random.randn(2*k, k)
    A = np.dot(pre_A.T, pre_A)
    (lam, Q) = np.linalg.eig(A)
    dynamics = get_rotation(k, angle, Q, P_inv=Q.T)
    seed = np.random.randn(k, 1)
    seed *= seed_factor / np.linalg.norm(seed)
    ds = [int(d) for d in ds.split()]
    loaders = get_lds_SCCAPMLs(
        num_data, 
        ds, 
        dynamics,
        seed=seed,
        lazy=lazy) 
    names = ['N', 'k', 'ds']
    vals = [
        str(num_data), 
        str(k),
        '-'.join([str(d) for d in ds])]
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
