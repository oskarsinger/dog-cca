import os
import experiments.plotting.filtering as epf
import experiments.plotting.correlation as epc

from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import get_nested_list_transpose as get_nlt

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.models import GridPlot

def plot_filtering_period_vs_component(
    model,
    width=1300,
    height=400,
    time_scale=3600,
    plot_path='.'):

    model_info = model.get_status()

    plots = [epf.plot_grouped_by_component(
                 m, 
                 historical=True, 
                 width=width, 
                 time_scale=time_scale)
             for m in model_info['learners']]

    return GridPlot(children=plots)

def plot_correlation_period_vs_component(
    model,
    width=1300,
    height=400,
    time_scale=3600,
    plot_path='.'):

    model_info = model.get_status()
    submodels = model_info['learners']

    plots = [epc.plot_grouped_by_component(
                 m, 
                 historical=True, 
                 width=width, 
                 time_scale=time_scale)
             for m in submodels]

    return GridPlot(children=plots)
