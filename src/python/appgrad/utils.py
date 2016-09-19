import numpy as np
import drrobert.debug as drdb

from numpy.random import choice
from linal.svd_funcs import get_svd_power
from drrobert.random import normal
from drrobert.misc import unzip

import global_utils as gu

# TODO: account for missing data to reduce computation
def get_gradients(Xs, basis_pairs):

    m = len(Xs)
    X_transforms = [(np.dot(X, unnormed), np.dot(X, normed))
                    for (X, (unnormed,normed)) in zip(Xs, basis_pairs)]

    for (unnormed, normed) in X_transforms:
        drdb.check_for_large_numbers(
            unnormed,
            'appgrad.utils get_gradients',
            'unnormed')
        drdb.check_for_large_numbers(
            normed,
            'appgrad.utils get_gradients',
            'normed')

    minus_term = sum(unzip(X_transforms)[1])

    drdb.check_for_large_numbers(
        minus_term,
        'appgrad.utils get_gradients',
        'minus_term')

    diffs = [(m-1)*unnormed - \
             (minus_term - normed)
             for (unnormed, normed) in X_transforms]

    for diff in diffs:
        drdb.check_for_large_numbers(
            diff,
            'appgrad.utils get_gradients',
            'diff')

    return [np.dot(X.T, diff) / X.shape[0]
            for (X, diff) in zip(Xs, diffs)]

def get_init_basis_pairs(Sxs, k):

    return [get_init_basis_pair(Sx, k)
            for Sx in Sxs]

def get_init_basis_pair(Sx, k):

    # Initialize unnormalized Gaussian matrix
    unn_Phi = normal(shape=(Sx.shape[0], k))

    # Normalize for initial normalized bases
    Phi = gu.misc.get_gram_normed(unn_Phi, Sx)

    return (Phi, Phi)
