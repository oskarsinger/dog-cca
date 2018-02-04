import click
import os

import numpy as np

from whitehorses.loaders.multiview import get_easy_SCCAPMLs
from drrobert.file_io import get_timestamped as get_ts

@click.command()
@click.option('--data-dir')
@click.option('--num-data', default=1000)
@click.option('--k', default=1)
@click.option('--ds', default='10 20 30')
@click.option('--lazy', default=True)
def run_things_all_day_bb(
    data_dir,
    num_data,
    k,
    ds,
    lazy):

    ds = [int(d) for d in ds.split()]
    loaders = get_easy_SCCAPMLs(num_data, k, ds, lazy=lazy) 
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
