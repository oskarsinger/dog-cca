import numpy as np

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

    if (S == 0).all():
        raise Exception('This should never be zero.')

    basis_quad = quad(unnormed, S)
    # print "Basis Quad", str((basis_quad == 0).all())
    normalizer = get_svd_power(basis_quad, -0.5)
    # print "Normalizer", str((normalizer == 0).all())

    return np.dot(unnormed, normalizer)
