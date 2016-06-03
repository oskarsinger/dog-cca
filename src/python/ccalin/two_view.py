import numpy as np

from data.servers.gram import BatchGramServer as BGS
from linal.gep.genelink import GenELinK as GLK
from linal.qr import get_q
from linal.utils import multi_dot

class CCALin:

    def __init__(self, 
        X_epsilon=10**(-5), Y_epsilon=10**(-5)):

        self.X_epsilon = X_epsilon
        self.Y_epsilon = Y_epsilon

    def fit(self, 
        X_ds, Y_ds, k,
        X_gs=None, Y_gs=None,
        max_iter=10000, 
        reg = 0.1,
        verbose=False):

        if X_gs is None:
            x_gs = BGS()

        if Y_gs is None:
            Y_gs = BGS()

        (X, Sx, Y, Sy) = ccau.init_data(X_ds, Y_ds, X_gs, Y_gs)
        A = self._get_A(X, Y)
        B = self._get_B(Sx, Sy)

        gep_solution = GLK().fit(
            A, B, 2k, 
            max_iter=max_iter, verbose=verbose)

        pre_Wx = gep_solution[:X_ds.cols(),:]
        pre_Wy = gep_solution[X_ds.cols():,:]
        U = np.random.randn(2k, k)
        pre_Wx = np.dot(pre_Wx, U)
        pre_Wy = np.dot(pre_Wy, U)
        Wx = get_q(
            pre_Wx, 
            inner_product=lambda x1, x2: multi_dot([x1, Sx, x2]))
        Wy = get_q(
            pre_Wy, 
            inner_product=lambda y1, y2: multi_dot([y1, Sy, y2]))

    def _get_A(self, X, Y):

        # Initialize cross-Gram matrix and A
        (n, pX) = X.shape
        size = pX + Y.shape[0]
        Sxy = np.dot(X.T, Y) + reg * np.identity(p)
        A = np.zeros((size, size))

        # Complete A
        A[:pX,pX:] += Sxy
        A[pX:,:pX] += Sxy.T

        return np.copy(A)

    def _get_B(self, Sx, Sy):

        # Initialize B
        pX = Sx.shape[0]
        size = pX + Sy.shape[0]
        B = np.zeros((size, size))
        
        # Complete B
        B[:pX,:pX] += Sx
        B[pX:,pX:] += Sy

        return np.copy(B)
