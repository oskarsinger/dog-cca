import numpy as np

import global_utils as gu

from linal.utils import multi_dot

def get_A(Xs):

    dims = [X.shape[1] for X in Xs]
    size = sum(dims)
    A = np.zeros((size, size))

    # Populate A matrix
    for i in range(len(Xs)):
        # Determine row range
        i_start = sum(dims[:i])
        i_end = sum(dims[:i+1])

        for j in range(i+1, len(Xs)):
            # Determine column range
            j_start = sum(dims[:j])
            j_end = sum(dims[:j+1])

            # Create cross-Gram matrix for i-th and j-th views
            Sxy = np.dot(Xs[i].T, Xs[j])

            # Insert cross-Gram matrix into A matrix
            A[i_start:i_end,j_start:j_end] += Sxy
            A[j_start:j_end,i_start:i_end] += Sxy.T

    return np.copy(A)

def get_B(Sxs):

    dims = [Sx.shape[0] for Sx in Sxs]
    size = sum(dims)
    B = np.zeros((size, size))

    # Populate B matrix
    for i in range(len(Sxs)):
        # Determine range
        begin = sum(dims[:i])
        end = sum(dims[:i+1])

        # Insert Gram matrix for i-th view into B matrix
        B[begin:end,begin:end] += Sxs[i]

    return np.copy(B)

def get_normed_Wxs(pre_Wxs, Sxs):

    # Perform generalized QR on each view's basis
    return [gu.misc.get_gram_normed(pre_Wx, Sx)
            for (pre_Wx, Sx) in zip(pre_Wxs, Sxs)]

def get_pre_Wxs(gep_solution, ds_list, k):

    # Set boundaries for extracting view-specific bases
    col_list = [ds.cols() for ds in ds_list]
    ends = [sum(col_list[:i+1])
            for i in range(len(col_list))]
    boundaries = zip([0]+ends[:-1], ends)

    # Create random projection
    U = np.random.randn(2*k, k)

    # Extract and project basis for each view
    return [np.dot(gep_solution[begin:end,:], U)
            for (begin,end) in boundaries]
