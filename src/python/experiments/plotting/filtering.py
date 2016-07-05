import os
import numpy as np

import global_utils as gu

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.palettes import Spectral11

def plot_grouped_by_component(
    model,
    historical=False,
    width=900,
    height=400,
    plot_path='.'):

    model_info = model.get_status()
    filtered_Xs = None

    if historical:
        filtered_Xs = model_info['filtering_history']
    else:
        filtered_Xs = _get_refiltered_Xs(model_info)

    k = model_info['k']
    seconds = model_info['ds_list'][0].dl.get_status()['seconds']
    num_rounds = model_info['num_rounds']
    X_axis = seconds * np.arange(filtered_Xs[0].shape[0])
    component_plots = []
    X_label = 'Time Step Observed'
    Y_label = 'Filtered Data Point for Component '

    for i in xrange(k):
        comp_map = {'Filtered X ' + str(j) + '\'s component ' :
                    (X_axis, X[:,i])
                    for (j, X) in enumerate(filtered_Xs)}
        component_plots.append(plot_lines(
            comp_map,
            X_label,
            Y_label + str(i),
            Y_label + str(i) + ' vs ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=width,
            height=height))

    filename = get_ts(
        'historical_' + str(historical) +
        '_component_grouped_' +
        '_cca_filtered_data_points_vs_time_step_observed') + '.html'
    filepath = os.path.join(plot_path, filename) 

    output_file(
        filepath, 
        'component-grouped CCA-filtered data points vs time step observed')
    show(vplot(*component_plots))

def plot_grouped_by_view(
    model,
    historical=False,
    width=900,
    height=400,
    plot_path='.'):

    model_info = model.get_status()
    filtered_Xs = None

    print 'Getting filtered data'
    if historical:
        filtered_Xs = model_info['filtering_history']
    else:
        filtered_Xs = _get_refiltered_Xs(model_info)

    k = model_info['k']
    seconds = model_info['ds_list'][0].dl.get_status()['seconds']
    num_rounds = model_info['num_rounds']
    X_plots = []
    X_label = 'Time Step Observed (seconds)'
    Y_label = 'Filtered Data Point for View '

    for (i, X) in enumerate(filtered_Xs):
        X_map = {'Filtered X dimension ' + str(j) : 
                 (seconds * np.arange(filtered_Xs[0].shape[0]), X[:,j])
                 for j in xrange(k)}
        X_plots.append(plot_lines(
            X_map,
            X_label,
            Y_label + str(i),
            Y_label + str(i) + ' vs ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=width,
            height=height))

    filename = get_ts(
        'historical_' + str(historical) +
        '_view_grouped_' +
        '_cca_filtered_data_points_vs_time_step_observed') + '.html'
    filepath = os.path.join(plot_path, filename) 

    output_file(
        filepath, 
        'view-grouped CCA-filtered data points vs time step observed')

    print 'Displaying plots'
    show(vplot(*X_plots))

def _get_refiltered_Xs(model_info):

    dss = model_info['ds_list']
    num_rounds = model_info['num_rounds']
    num_views = model_info['num_views']
    Phis = model_info['bases']
    filtered_Xs = None

    if num_rounds > 1:
        for ds in dss:
            ds.refresh()
    
        for i in range(num_rounds):
            Xs = [ds.get_data() for ds in dss]

            if filtered_Xs is None:
                filtered_Xs = [np.dot(X[-1,:], Phi)
                               for (X, Phi) in zip(Xs, Phis)]
            else:
                for j in xrange(num_views):
                    current = filtered_Xs[j]
                    new = np.dot(Xs[j][-1,:], Phis[j])
                    filtered_Xs[j] = np.vstack([current, new])

    else:
        gss = model_info['gs_list']
        Xs = gu.data.init_data(dss, gss)[0]
        filtered_Xs = [np.dot(X, Phi) 
                       for (X, Phi) in zip(Xs, Phis)]

    return [np.copy(fX) for fX in filtered_Xs]
