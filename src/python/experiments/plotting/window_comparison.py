import os
import experiments.plotting.filtering as epf

from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import get_nested_list_transpose as get_nlt

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.models import GridPlot

def plot_component_vs_period(
    model,
    width=1300,
    height=400,
    time_scale=3600,
    plot_path='.'):

    model_info = model.get_status()
    submodels = model_info['learners']

    plots = [epf.plot_grouped_by_component(
                 m, 
                 historical=True, 
                 width=width, 
                 time_scale=time_scale)
             for m in submodels]

    return GridPlot(children=plots)#get_nlt(plots))
