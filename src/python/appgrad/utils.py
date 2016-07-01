import numpy as np

from numpy.random import choice
from linal.utils import quadratic as quad
from linal.svd_funcs import get_svd_power
from drrobert.random import normal
from drrobert.misc import unzip

def get_gradients(Xs, basis_pairs):

    m = len(Xs)
    X_transforms = [(np.dot(X, unnormed), np.dot(X, normed))
                    for (X, (unnormed,normed)) in zip(Xs, basis_pairs)]
    minus_term = sum(unzip(X_transforms)[1])
    diffs = [(m-1)*X_transforms[i][0] - \
             (minus_term - X_transforms[i][1])
             for i in range(m)]

    return [np.dot(X.T, diff) / X.shape[0]
            for (X, diff) in zip(Xs, diffs)]

def get_init_basis_pairs(Sxs, k):

    return [get_init_basis_pair(Sx, k)
            for Sx in Sxs]

def get_init_basis_pair(Sx, k):

    # Initialize unnormalized Gaussian matrix
    unn_Phi = normal(shape=(Sx.shape[0], k), scale=1000)

    # Normalize for initial normalized bases
    Phi = get_gram_normed(unn_Phi, Sx)

    return (Phi, Phi)
