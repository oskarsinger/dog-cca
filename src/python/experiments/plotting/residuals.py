import os

import numpy as np

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.palettes import Spectral11

def plot_grouped_by_component(
    model,
    width=900,
    height=400,
    plot_path='.'):

    model_info = model.get_status()
    residuals = model_info['residuals']
    seconds = model_info['ds_list'][0].dl.get_status()['seconds']
    num_rounds = model_info['num_rounds']
    X_axis = seconds * np.arange(num_rounds)
    residual_plots = []
    X_label = 'Time Step Observed'
    Y_label = 'Residuals for Component '

    for i in xrange(k):
        comp_map = {_get_data_map_key(fs) : (X_axis, r[:,i])
                    for (fs, r) in residuals.items()}
        component_plots.append(plot_lines(
            comp_map,
            X_label,
            Y_label + str(i),
            Y_label + str(i) + ' vs ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=width,
            height=height))

    filename = get_ts(
        'component_grouped_' +
        '_cca_filtered_data_point_residuals_vs_time_step_observed' + '.html')
    filpath = os.path.join(plot_path, filename)
    output_file(
        filepath,
        'component-grouped CCA-filtered data point residuals vs time step observed')
    show(vplot(*component_plots))

def _get_data_map_key(fs):
    
    l = list(fs)

    return ' '.join([
        'Residuals for view',
        str(l[0]),
        'vs view',
        str(l[1])])
