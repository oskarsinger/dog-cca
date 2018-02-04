import click
import os

import numpy as np

from whitehorses.loaders.multiview.cca import get_SCP_SCCAPMLs
from drrobert.file_io import get_timestamped as get_ts
from theline.utils import get_quadratic, get_rotation

@click.command()
@click.option('--data-dir')
@click.option('--num-data', default=1000)
@click.option('--k', default=2)
@click.option('--ds', default=' '.join([str(10 * i) for i in range(1, 100)]))
@click.option('--rho', 0.8)
@click.option('--lazy', default=True)
def run_things_all_day_bb(
    data_dir,
    num_data,
    k,
    ds,
    rho,
    lazy):

    pass

if __name__=='__main__':
    run_things_all_day_bb()
