import os

import numpy as np
import global_utils as gu
import utils as epu

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts

from bokeh.plotting import figure, output_file
from bokeh.palettes import Spectral11

def plot_grouped_by_component(
    model,
    historical=False,
    width=1200,
    height=400,
    time_scale=24*3600,
    plot_path='.'):

    model_info = model.get_status()
    filtered_Xs = None

    print 'Getting filtered data'

    if historical:
        filtered_Xs = model_info['filtering_history']
    else:
        filtered_Xs = epu.get_refiltered_Xs(model_info)

    ds2name = lambda ds: ds.get_status()['data_loader'].name()
    names = [ds2name(ds) + ' (' + str(i) + ')'
             for (i, ds) in enumerate(model_info['ds_list'])]
    name_and_data = zip(names, filtered_Xs)
    k = model_info['k']
    num_rounds = model_info['num_rounds']
    X_axis = epu.get_X_axis(
        model_info, filtered_Xs[0].shape[0], time_scale)
    component_plots = []
    X_label = 'Time Step Observed (days)'
    Y_label = 'Filtered Data Point for Component '

    for i in xrange(k):
        comp_map = {'Filtered ' + name + '\'s component ' :
                    (X_axis, X[:,i])
                    for (name, X) in name_and_data}
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

    return component_plots

def plot_grouped_by_view(
    model,
    historical=False,
    width=1200,
    height=400,
    time_scale=24*3600,
    plot_path='.'):

    model_info = model.get_status()
    filtered_Xs = None

    print 'Getting filtered data'
    if historical:
        filtered_Xs = model_info['filtering_history']
    else:
        filtered_Xs = epu.get_refiltered_Xs(model_info)

    ds2name = lambda ds: ds.get_status()['data_loader'].name()
    names = [ds2name(ds) + ' (' + str(i) + ')'
             for (i, ds) in enumerate(model_info['ds_list'])]
    name_and_data = zip(names,filtered_Xs)
    k = model_info['k']
    num_rounds = model_info['num_rounds']
    X_plots = []
    X_axis = epu.get_X_axis(
        model_info, filtered_Xs[0].shape[0], time_scale)
    X_label = 'Time Step Observed (days)'
    Y_label = 'Filtered Data Points for View '

    for (name, X) in name_and_data:
        X_map = {'Filtered ' + name + ' dimension ' + str(j) : 
                 (X_axis, X[:,j])
                 for j in xrange(k)}
        X_plots.append(plot_lines(
            X_map,
            X_label,
            Y_label + name,
            Y_label + name + ' vs ' + X_label,
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
    return X_plots
