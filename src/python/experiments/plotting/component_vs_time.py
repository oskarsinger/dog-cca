import os

import numpy as np
import global_utils as gu
import utils as epu

from lazyprojector import plot_matrix_heat
from drrobert.file_io import get_timestamped as get_ts

from bokeh.plotting import figure, output_file
from bokeh.palettes import BuPu9

def plot_components_over_time(
    model_info,
    width=1200,
    height=400,
    time_scale=24*3600,
    datetime_axis=False,
    plot_path='.'):

    basis_history = model_info['basis_history']

    names = epu.get_loader_names(model_info)
    ns_and_Phis = zip(names, basis_history)
    k = model_info['k']
    component_plots = []
    color_scheme = list(reversed(BuPu9))
    x_name = 'Timestep'
    y_name = 'Ambient Coordinate'
    val_name = 'Energy'

    for i in xrange(k):
        value_matrix = np.abs(basis_history[i])
        y_labels = [str(j) for j in xrange(value_matrix.shape[0])]
        x_labels = ['Fill this in with the proper sequence of datetimes.']

        component_plots.append(plot_matrix_heat(
            value_matrix,
            x_labels, y_labels,
            'Component estimates vs time',
            x_name, y_name, val_name,
            color_scheme=color_scheme))

    # TODO: set plot output file, get x labels correct, 

    return component_plots
