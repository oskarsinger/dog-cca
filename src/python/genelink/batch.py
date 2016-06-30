import numpy as np

from linal.qr import get_q
from linal.utils import multi_dot, quadratic as quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

class BatchGenELinKSolver:

    def __init__(self, 
        k, A, B,
        epsilon=10**(-5),
        get_optimizer=None):

        # Set the easy ones
        self.k = k
        self.epsilon = epsilon

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

        # Verify and set the optimizer factory
        if get_optimizer is None:
            get_optimizer = MAG

        self.get_optimizer = get_optimizer

        # Initialize object-wide state variable for W
        self.W = None

    def fit(self, 
        eta=0.1,
        max_iter=1000, 
        verbose=False):

        if optimizer is None:
            optimizer = MAG()

        inner_prod = lambda x,y: multi_dot([x, self.B, y])

        # Initialize iteration variables
        W_t = get_q(np.random.randn(d, k), inner_prod=inner_prod)
        W_t1 = None
        t = 0

        while not converged and t < max_iter:

            # Compute initialization for trace minimization
            B_term = get_svd_power(quad(W, self.B), power=-1)
            A_term = quad(W, self.A)
            init = multi_dot([W_t, B_term, A_term])

            # Get (t+1)-th iterate of W
            unn_W_t1 = self._get_new_W(init, verbose)
            W_t1 = "Put in the GenELinKSubroutine later"

            # Check for convergence
            converged = is_converged(
                W_t, W_t1, self.epsilon, verbose)

            # Update iteration variables
            W_t = np.copy(W_t1)
            t += 1

        self.W = np.copy(W_t)

    def get_basis(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return np.copy(self.W)

    def _get_objective(self, W):

        inner = quad(W, 0.5*self.B - self.A)

        return np.trace(inner)
