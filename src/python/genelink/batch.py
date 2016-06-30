import numpy as np

from linal.qr import get_q
from linal.utils import multi_dot
from linal.utils import quadratic as quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

class BatchGenELinKSolver:

    def __init__(self, 
        k, A, B,
        epsilon=10**(-5)):

        self.A = A
        self.B = B
        self.epsilon = epsilon
        self.W = None

    def fit(self, 
        eta=0.1,
        optimizer=None,
        max_iter=1000, 
        verbose=False):

        (nA, pA) = self.A.shape
        (nB, pB) = self.B.shape
        ds = [nA, pA, nB, pB]

        if not all([d == nA for d in ds]):
            raise ValueError(
                'All dimensions of both A and B should be equal.')

        if optimizer is None:
            optimizer = MAG()

        d = nA
        inner_prod = lambda x,y: multi_dot([x, self.B, y])

        # Initialize iteration variables
        W_t = get_q(np.random.randn(d, k), inner_prod=inner_prod)
        W_t1 = None
        i = 0

        while not converged and i < max_iter:

            # Compute initialization for trace minimization
            B_term = get_svd_power(quad(W, self.B), power=-1)
            A_term = quad(W, self.A)
            init = multi_dot([W_t, B_term, A_term])

            # Get (t+1)-th iterate of W
            unn_W_t1 = self._get_new_W(init, verbose)
            W_t1 = get_q(unn_W_t1, inner_prod=inner_prod)

            # Check for convergence
            converged = is_converged(
                W_t, W_t1, self.epsilon, verbose)

            # Update iteration variables
            W_t = np.copy(W_t1)
            i += 1

        self.W = np.copy(W_t)

    def get_basis(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return np.copy(self.W)

    def _get_new_W(self, init, eta, optimizer, verbose):

        # Initialize iteration variables
        W_i = init 
        W_i1 = None
        i = 1

        while not converged:
            # Update iteration variable
            eta_i = eta / i**(0.5)

            # Get new parameter estimate
            gradient = np.dot((0.5*self.B - self.A).T, W_i)
            W_i1 = optimizer.get_update(W_i, gradient, eta_t)

            # Check for convergence
            converged = is_converged(W_i, W_i1, self.epsilon, verbose)

            # Update iteration variables
            t = np.copy(W_i1)

        return np.copy(W_i)

    def _get_objective(self, W):

        inner = quad(W, 0.5*self.B - self.A)

        return np.trace(inner)
