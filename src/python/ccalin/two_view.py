import numpy as np

from data.servers.gram import BatchGramServer as BGS
from linal.gep.genelink import GenELinK as GLK
from linal.qr import get_q
from linal.utils import multi_dot

class CCALin:

    def __init__(self, 
        k,
        X_epsilon=10**(-5), Y_epsilon=10**(-5)):

        self.k = k
        self.X_epsilon = X_epsilon
        self.Y_epsilon = Y_epsilon

        self.has_been_fit = False
        self.Phi = None
        self.Psi = None

    def fit(self, 
        X_ds, Y_ds,
        X_gs=None, Y_gs=None,
        max_iter=10000, 
        reg=0.1,
        verbose=False):

        if X_gs is None:
            X_gs = BGS()

        if Y_gs is None:
            Y_gs = BGS()

        # Initialize the data and Gram matrices
        (Xs, Sxs) = ccau.init_data([X_ds, Y_ds], [X_gs, Y_gs])
        (X, Y, Sx, Sy) = (Xs[0], Xs[1], Sxs[0], Sxs[1])

        # Prepare the GEP input
        A = self._get_A(X, Y)
        B = self._get_B(Sx, Sy)

        # Get the GEP solution
        gep_solution = GLK().fit(
            A, B, 2*self.k, 
            max_iter=max_iter, verbose=verbose)

        # Extract the unnormalized CCA solution from the GEP solution
        pre_Wx = gep_solution[:X_ds.cols(),:]
        pre_Wy = gep_solution[X_ds.cols():,:]
        U = np.random.randn(2*self.k, self.k)
        pre_Wx = np.dot(pre_Wx, U)
        pre_Wy = np.dot(pre_Wy, U)

        # Normalize the CCA solution
        self.Phi = get_q(
            pre_Wx, 
            inner_product=lambda x1, x2: multi_dot([x1, Sx, x2]))
        self.Psi = get_q(
            pre_Wy, 
            inner_product=lambda y1, y2: multi_dot([y1, Sy, y2]))

        # Update model state
        self.has_been_fit = True

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return (np.copy(self.Phi), np.copy(self.Psi))

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
