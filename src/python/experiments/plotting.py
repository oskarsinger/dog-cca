import os

import numpy as np

from math import log

from lazyprojector import plot_matrix_heat, plot_lines
from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import unzip

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.palettes import GnBu9, YlOrRd9, BuPu9

def plot_label_counts(data_dir, plot_path='.'):

    labels = None
    raise Exception('Implement label extraction.')
    counts = {}

    for label in labels:
        if label not in counts:
            counts[label] = 1 
        else:
            counts[label] += 1

    factors = counts.keys()
    x = counts.values()
    x_max = max(x)
    scale = 10**(log(x_max, base=10))
    x_max = (x_max - x_max % scale) + scale
    p = figure(title="Label Counts", tools="resize,save", y_range=factors, x_range=[0,x_max])

    p.segment(0, factors, x, factors, line_width=2, line_color="green", )
    p.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

    name = get_ts('label_counts') + '.html'
    filepath = os.path.join(plot_path, name)
    
    output_file(filepath, title="label counts")
    show(p)

def plot_cca_filtering(
    filtered_X,
    filtered_Y,
    X_width=900,
    X_height=400,
    Y_width=900,
    Y_hieght=400,
    plot_path='.'):

    (n1, k1) = filtered_X.shape
    (n2, k2) = filtered_Y.shape

    if (not n1 == n2) or (not k1 == k2):
        raise ValueError(
            'filtered_X and filtered_Y should have the same dimensions.')

    (n, k) = (n1, k1)
    X_map = {'X filter dimension ' + str(i) : filtered_X[:,i]
             for i in xrange(k)}
    Y_map = {'Y filter dimension ' + str(i) : filtered_Y[:,i]
             for i in xrange(k)}
    X_plot = plot_lines(
        X_map,
        "Time Step",
        "Filtered X Data Point",
        "Filtered X Data Points vs Time Step Observed",
        color=YlGnBu9,
        width=X_width,
        heigh=X_height)

    Y_plot = plot_lines(
        Y_map,
        "Time Step",
        "Filtered Y Data Point",
        "Filtered Y Data Points vs Time Step Observed",
        color=YlOrRd9,
        width=Y_width,
        height=Y_height)

    filename = get_ts(str(n) + '_n_' + str(k) + '_k_' +
        'cca_filtered_data_points_vs_time_step_observed') + '.html'
    filepath = os.path.join(plot_path, filename) 

    output_file(
        filepath, 
        'CCA-filtered data points vs time step observed')
    show(vplot(X_plot, Y_plot))

def plot_canonical_bases(Phis, Psi=None, plot_path='.'):

    (ps, ks) = unzip([Phi.shape for Phi in Phis])
    (p_Psi, k_Psi) = Psi.shape
    k = None
    
    if len(set(ks + [k_Psi])) == 1:
        k = k_Psi
    else:
        raise ValueError(
            'Second dimension of each basis should be equal.')

    Phis_features = [[str(i) for i in range(p)]
                     for p in ps]
    Psi_features = [str(i) for i in range(p_Psi)]
    basis_elements = [str(i) for i in range(k)]

    Phis_ps = [_plot_basis(Phi, 'Phi' + str(i), f, basis_elements)
               for i, (Phi, f) in enumerate(zip(Phis, Phis_features))]
    Psi_p = _plot_basis(Psi, 'Psi', Psi_features, basis_elements)
    
    prefix = str(k) + '_k_' + \
        '_'.join([str(p) for p in ps]) + '_phis_'
    filename = get_ts(prefix + 
        'mass_per_feature_over_bases_matrix_heat_plot') + '.html'
    filepath = os.path.join(plot_path, filename)
    plot_list = Phis_ps + [Psi_p]

    output_file(
        filepath, 
        'percent mass per feature over bases')
    show(vplot(*plot_list))

def _plot_basis(basis, name, features, basis_elements):

    return plot_matrix_heat(
        np.abs(basis),
        basis_elements,
        features,
        'Percent mass per feature over ' + name + ' basis elements',
        name + ' basis element',
        'feature',
        'mass',
        color_scheme=list(reversed(BuPu9)))
