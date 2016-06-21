import os

import numpy as np

from math import log

from lazyprojector import plot_matrix_heat, plot_lines
from drrobert.file_io import get_timestamped as get_ts
from drrobert.misc import unzip

from bokeh.plotting import figure, show, output_file, vplot
from bokeh.palettes import GnBu9, YlOrRd9, BuPu9

def plot_canonical_bases(Phis, Psi=None, plot_path='.'):

    (ps, ks) = unzip([Phi.shape for Phi in Phis])
    (p_Psi, k_Psi) = (0, 0)

    if Psi is not None:
        p_Psi, k_Psi = Psi.shape
        ks = ks + [k_Psi]

    k = None
    
    if len(set(ks)) == 1:
        k = ks[0]
    else:
        raise ValueError(
            'Second dimension of each basis should be equal.')

    Phis_features = [[str(i) for i in range(p)]
                     for p in ps]
    basis_elements = [str(i) for i in range(k)]

    Phis_ps = [_plot_basis(Phi, 'Phi' + str(i), f, basis_elements)
               for i, (Phi, f) in enumerate(zip(Phis, Phis_features))]
    Psi_p = None

    if Psi is not None:
        Psi_features = [str(i) for i in range(p_Psi)]
        Psi_p = _plot_basis(Psi, 'Psi', Psi_features, basis_elements)
    
    prefix = str(k) + '_k_' + \
        '_'.join([str(p) for p in ps]) + '_phis_'
    filename = get_ts(prefix + 
        'mass_per_feature_over_bases_matrix_heat_plot') + '.html'
    filepath = os.path.join(plot_path, filename)

    if Psi is not None:
        Phis_ps = Phis_ps + [Psi_p]

    output_file(
        filepath, 
        'percent mass per feature over bases')
    show(vplot(*Phis_ps))

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
