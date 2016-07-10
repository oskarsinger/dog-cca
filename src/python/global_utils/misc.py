import numpy as np

from optimization.utils import is_converged as is_conv
from linal.utils import quadratic as quad
from linal.svd_funcs import get_svd_power

def is_converged(
    unn_Phi_pairs,
    epsilons, 
    verbose):

    conv_info = zip(unn_Phi_pairs, epsilons)

    return [is_conv(unn_Phi_t, unn_Phi_t1, eps, verbose)
            for (unn_Phi_t, unn_Phi_t1), eps in conv_info]

def check_k(ds_list, k):

    if not gu.misc.is_k_valid(ds_list, k):
        raise ValueError(
            'Parameter k must be <= minimum column dimension among views.')

def is_k_valid(ds_list, k):

    p = min([ds.cols() for ds in ds_list])

    return k <= p

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

def get_gram_normed(unnormed, S):

    basis_quad = quad(unnormed, S)
    normalizer = get_svd_power(basis_quad, -0.5)

    return np.dot(unnormed, normalizer)
