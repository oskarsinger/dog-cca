import numpy as np

from numpy.random import choice
from linal.utils import quadratic as quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged as is_conv
from drrobert.random import normal
from drrobert.misc import unzip

def is_k_valid(ds_list, k):

    p = min([ds.cols() for ds in ds_list])

    return k <= p

def is_converged(
    unn_Phi_pairs,
    epsilons, 
    verbose):

    conv_info = zip(unn_Phi_pairs, epsilons)

    return [is_conv(unn_Phi_t, unn_Phi_t1, eps, verbose)
            for (unn_Phi_t, unn_Phi_t1), eps in conv_info]

def get_objective(Xs, Phis):

    if not len(Xs) == len(Phis):
        raise ValueError(
            'Xs and Phis should have the same number of elements.')

    X_transforms = [np.dot(X, Phi)
                    for X, Phi in zip(Xs, Phis)]
    diffs = [X_transforms[i] - X_transforms[j]
             for i in range(len(Xs)) 
             for j in range(i+1,len(Xs))]
    residuals = [np.linalg.norm(diff) for diff in diffs]

    return sum(residuals)

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

def get_gram_normed(unnormed, S):

    if (S == 0).all():
        raise Exception('This should never be zero.')

    basis_quad = quad(unnormed, S)
    # print "Basis Quad", str((basis_quad == 0).all())
    normalizer = get_svd_power(basis_quad, -0.5)
    # print "Normalizer", str((normalizer == 0).all())

    return np.dot(unnormed, normalizer)
