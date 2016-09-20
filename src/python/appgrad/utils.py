import numpy as np
import drrobert.debug as drdb

from numpy.random import choice
from linal.svd_funcs import get_svd_power
from drrobert.random import normal
from drrobert.misc import unzip

import global_utils as gu

def get_gradients(Xs, basis_pairs):

    m = len(Xs)
    X_transforms = [(np.dot(X, unnormed), np.dot(X, normed))
                    for (X, (unnormed,normed)) in zip(Xs, basis_pairs)]

    info = zip(
        Xs, basis_pairs, X_transforms)
    for thang in info:
        X = thang[0]
        (unnormed_Phi, normed_Phi) = thang[1]
        (unnormed_Phi_X, normed_Phi_X) = thang[2]

        print 'X', X
        print 'unnormed_Phi', unnormed_Phi
        print 'normed_Phi', normed_Phi
        print 'unnormed_Phi_X', unnormed_Phi_X
        print 'normed_Phi_X', normed_Phi_X

        """
        drdb.check_for_large_numbers(
            unnormed_Phi_X,
            'appgrad.utils get_gradients',
            'unnormed_Phi_X')
        """
        drdb.check_for_large_numbers(
            normed_Phi_X,
            'appgrad.utils get_gradients',
            'normed_Phi_X')

    minus_term = sum(unzip(X_transforms)[0])

    """
    drdb.check_for_large_numbers(
        minus_term,
        'appgrad.utils get_gradients',
        'minus_term')
    """

    diffs = [(m-1)*unnormed - \
             (minus_term - unnormed)
             for (unnormed, normed) in X_transforms]

    for diff in diffs:
        print diff
        drdb.check_for_large_numbers(
            diff,
            'appgrad.utils get_gradients',
            'diff')

    gradients = [np.dot(X.T, diff) / X.shape[0]
                 for (X, diff) in zip(Xs, diffs)]

    for g in gradients:
        drdb.check_for_large_numbers(
            g,
            'AGNVCCA _get_basis_updates at round ' + str(self.num_rounds),
            'gradient')

    return gradients

def get_init_basis_pairs(Sxs, k):

    return [get_init_basis_pair(Sx, k)
            for Sx in Sxs]

def get_init_basis_pair(Sx, k):

    # Initialize unnormalized Gaussian matrix
    unn_Phi = normal(shape=(Sx.shape[0], k))
    
    print 'unn_Phi', unn_Phi

    # Normalize for initial normalized bases
    Phi = gu.misc.get_gram_normed(unn_Phi, Sx)

    print 'Phi', Phi

    return (unn_Phi, Phi)
