import os

import numpy as np

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.palettes import Spectral11

def plot_cca_filtering(
    model,
    X_width=900,
    X_height=400,
    Y_width=900,
    Y_hieght=400,
    plot_path='.'):

    filtered_Xs = model.get_status()['filtering_history']
    shapes = [X.shape for X in filtered_Xs] 
    k = None

    if len(set([shape[1] for shape in shapes])) == 1:
        k = shapes[0][1]
    else:
        raise ValueError(
            'Filter sequences should all have same first dimension.')

    X_plots = []
    X_label = 'Time Step Observed'
    Y_label = 'Filtered Data Point for View '

    for (i, X) in enumerate(filtered_Xs):
        X_map = {'X filter dimension ' + str(i) : 
                 (np.arange(X.shape[0]) ,X[:,i])
                 for i in xrange(k)}
        X_plots.append(plot_lines(
            X_map,
            X_label,
            Y_label + str(i),
            Y_label + str(i) + ' vs ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=X_width,
            height=X_height))

    filename = get_ts(
        'cca_filtered_data_points_vs_time_step_observed') + '.html'
    filepath = os.path.join(plot_path, filename) 

    output_file(
        filepath, 
        'CCA-filtered data points vs time step observed')
    show(vplot(*X_plots))

