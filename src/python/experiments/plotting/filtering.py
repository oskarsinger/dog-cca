import os

import numpy as np
import global_utils as gu
import utils as epu

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts
from linal.utils import get_thresholded as get_thresh

from bokeh.plotting import figure, output_file
from bokeh.palettes import Spectral11

def plot_every_basis_all_data_grouped_by_component(
    model_info,
    width=1200,
    height=400,
    time_scale=24*3600,
    upper=1.0, lower=-1.0,
    datetime_axis=False,
    plot_path='.'):

    plots = []

    for learner in model_info['learners']:
        submodel_info = learner.get_status()
        model_info['bases'] = submodel_info['bases']
        plots.extend(
            plot_grouped_by_component(
                model_info,
                width=width,
                height=height,
                time_scale=time_scale,
                upper=upper, lower=lower,
                datetime_axis=datetime_axis,
                plot_path=plot_path))

    return plots

def plot_grouped_by_component(
    model_info,
    historical=False,
    absval=False,
    width=1200,
    height=400,
    time_scale=24*3600,
    upper=1.0, lower=-1.0,
    datetime_axis=False,
    plot_path='.'):

    filtered_Xs = None

    print 'Getting filtered data'

    if historical:
        filtered_Xs = model_info['filtering_history']
    else:
        filtered_Xs = epu.get_refiltered_Xs(model_info)

    if absval:
        filtered_Xs = [np.abs(fX) for fX in filtered_Xs]

    filtered_Xs = [get_thresh(fX, upper=upper, lower=lower)
                   for fX in filtered_Xs]
    names = epu.get_loader_names(model_info)
    ns_and_Xs = zip(names, filtered_Xs)
    k = model_info['k']
    X_axis = epu.get_filtering_X_axis(
        model_info, 
        filtered_Xs[0].shape[0], 
        time_scale=time_scale, 
        datetime_axis=datetime_axis)
    component_plots = []
    X_label = 'Time (days)'
    Y_label = 'Component'
    gos = lambda j: _get_offset(j, upper, lower)

    for i in xrange(k):
        comp_map = {name + '\'s component ' :
                    (X_axis, X[:,i] + gos(j))
                    for j, (name, X) in enumerate(ns_and_Xs)}
        component_plots.append(plot_lines(
            comp_map,
            X_label,
            Y_label + ' ' + str(i),
            Y_label + ' ' + str(i) + ' vs ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=width,
            height=height))

    if datetime_axis:
        for plot in component_plots:
            epu.set_datetime_xaxis(plot)

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
    model_info,
    historical=False,
    width=1200,
    height=400,
    time_scale=24*3600,
    upper=1.0, lower=-1.0,
    datetime_axis=False,
    plot_path='.'):

    filtered_Xs = None

    print 'Getting filtered data'
    if historical:
        filtered_Xs = model_info['filtering_history']
    else:
        filtered_Xs = epu.get_refiltered_Xs(model_info)

    filtered_Xs = [get_thresh(fX, upper=upper, lower=lower)
                   for fX in filtered_Xs]
    names = epu.get_loader_names(model_info)
    k = model_info['k']
    X_plots = []
    X_axis = epu.get_filtering_X_axis(
        model_info, filtered_Xs[0].shape[0], 
        time_scale=time_scale, datetime_axis=datetime_axis)
    X_label = 'Time (days)'
    Y_label = 'Components of '
    gos = lambda j: _get_offset(j, upper, lower)

    for (name, X) in zip(names,filtered_Xs):
        X_map = {name + ' component ' + str(j) : 
                 (X_axis, X[:,j] + gos(j))
                 for j in xrange(k)}
        X_plots.append(plot_lines(
            X_map,
            X_label,
            Y_label + name,
            Y_label + name + ' vs ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=width,
            height=height))

    if datetime_axis:
        for plot in X_plots:
            epu.set_datetime_xaxis(plot)

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

def _get_offset(count, upper, lower):

    offset = 0

    if upper is not None and lower is not None:
        cushion = 0.05 * (upper - lower)
        offset = (cushion + upper - lower) * (count - 1)

    return offset
