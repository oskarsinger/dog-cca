import os

import numpy as np
import global_utils as gu
import utils as epu

from lazyprojector import plot_lines
from drrobert.file_io import get_timestamped as get_ts
from linal.utils import get_thresholded as get_thresh

from bokeh.plotting import figure, output_file
from bokeh.palettes import Spectral11

def plot_grouped_by_component(
    model,
    historical=False,
    width=1200,
    height=400,
    time_scale=24*3600,
    upper=1.0, lower=-1.0,
    datetime_axis=False,
    plot_path='.'):
    
    model_info = model.get_status()
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
    num_views = model_info['num_views']
    X_axis = epu.get_filtering_X_axis(
        model_info, filtered_Xs[0].shape[0], 
        time_scale=time_scale, datetime_axis=datetime_axis)
    component_plots = []
    X_label = 'Time (days)'
    Y_label = 'Correlation'
    goc = lambda i, j, l: _get_offset_correlation(
        filtered_Xs, num_views, i, j, l, upper, lower)

    for i in xrange(k):
        comp_map = {'Correlation of ' + \
                        names[j] + ' and ' + \
                        names[l] :
                    (X_axis, goc(i, j, l))
                    for j in xrange(num_views)
                    for l in xrange(j+1, num_views)} 

        component_plots.append(plot_lines(
            comp_map,
            X_label,
            Y_label + ' ' + str(i),
            Y_label + ' ' + str(i) + ' and ' + X_label,
            colors=Spectral11[:4]+Spectral11[-4:],
            width=width,
            height=height))

    if datetime_axis:
        for plot in component_plots:
            epu.set_datetime_xaxis(plot)

    filename = get_ts(
        'historical_' + str(historical) +
        '_component_grouped_' +
        '_canonical_correlations_vs_time_step_observed') + '.html'
    filepath = os.path.join(plot_path, filename)

    output_file(
        filepath,
        'component-grouped Canonical correlations vs time step observed')

    return component_plots

def _get_offset_correlation(fXs, num_views, i, j, l, upper, lower):

    correlation = fXs[j][:,i] * fXs[l][:,i]
    cushion = 0.05 * (upper - lower)
    count = j * num_views + l - (j * (j - 1) / 2 + j)
    offset = (cushion + upper - lower) * (count - 1)

    return correlation + offset
