import numpy as np

from linal.utils import get_quadratic as get_quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from genelink import GenELinKSubroutine

class BatchGenELinKSolver:

    def __init__(self, 
        k, A, B,
        epsilon=10**(-5),
        get_optimizer=None,
        max_iter=1000,
        subroutine_max_iter=100,
        verbose=False):

        # Set the easy ones
        self.k = k
        self.epsilon = epsilon
        self.get_optimizer = get_optimizer
        self.max_iter = max_iter
        self.verbose = verbose

        # Check that dimensions of A and B are valid
        (nA, pA) = A.shape
        (nB, pB) = B.shape
        ds = [nA, pA, nB, pB]

        if not len(set(ds)) == 1:
            raise ValueError(
                'All dimensions of both A and B should be equal.')

        self.A = A
        self.B = B
        self.d = ds[0]

        # Initialize GenELinK subroutine
        self.subroutine = GenELinKSubroutine(
                self.k, self.d, 
                get_optimizer=get_optimizer,
                max_iter=subroutine_max_iter,
                verbose=self.verbose)

        # Initialize object-wide state variable for W
        self.W = None

        self.has_been_fit = False

    def fit(self, eta=0.1):

        # Initialize iteration variables
        converged = False
        W_t = None
        W_t1 = None
        t = 0

        while not converged and t < self.max_iter:

            if W_t is None:
                (W_t, sub_converged) = self.subroutine.get_update(
                    self.A, self.B, eta)
            else:
                (W_t1, sub_converged) = self.subroutine.get_update(
                    self.A, self.B, eta)

                # Check for convergence
                converged = is_converged(
                    W_t, W_t1, self.epsilon, self.verbose)

                # Update W_t
                W_t = np.copy(W_t1)

            # Update number of iterations
            t += 1

        self.W = np.copy(W_t)

        self.has_been_fit = True

    def get_basis(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return np.copy(self.W)

    def _get_objective(self, W):

        inner = get_quad(W, 0.5*self.B - self.A)

        return np.trace(inner)
