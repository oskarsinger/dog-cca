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
    residual_plots = []
    X_label = 'Time Step Observed'
    Y_label = lambda i,j: ' '.join([
        'Residual for View',
        str(i),
        'vs View',
        str(j)])
