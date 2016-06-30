import numpy as np

from linal.qr import get_q
from linal.utils import multi_dot, quadratic as quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from optimization.optimizers import AbstractOptimizer

class GenELinKSubroutine:

    def __init__(self,
        k, d,
        epsilon=10**(-3),
        get_optimizer=None):

        # Set the easy ones
        self.k = k
        self.d = d
        self.epsilon = epsilon

        # Verify and set the optimizer factory
        if get_optimizer is None:
            get_optimizer = MAG

        self.get_optimizer = get_optimizer

        # Initialize object-wide state variable for W
        self.W = None

    def get_update(self, A, B, eta):

        (nA, pA) = self.A.shape
        (nB, pB) = self.B.shape
        ds = [nA, pA, nB, pB]

        if not len(set(ds + [self.d])) == 1:
            raise ValueError(
                'All dimensions of both A and B should be equal.')

        # Initializing optimizer for this round
        optimizer = self.get_optimizer()
        
        # Create inner product for Gram-Schmidt orthogonalization
        inner_prod = lambda x,y: multi_dot([x, B, y])

        if self.W is None:
            unnormed = np.random.randn(self.d, self.k)
            self.W = get_q(unnormed, inner_product=inner_product)

        # Initialize iteration variables
        B_term = get_svd_power(quad(W, B), power=-1)
        A_term = quad(W, A)
        W_i = multi_dot([self.W, B_term, A_term])
        W_i1 = None
        i = 1

        while not converged:
            # Update iteration variable
            eta_i = eta / i**(0.5)

            # Get new parameter estimate
            gradient = np.dot((0.5*B - A).T, W_i)
            W_i1 = optimizer.get_update(W_i, gradient, eta_i)

            # Check for convergence
            converged = is_converged(W_i, W_i1, self.epsilon, verbose)

            # Update iteration variables
            W_i = np.copy(W_i1)

        # Update global state of W
        self.W = get_q(W_i, inner_product=inner_product)

        return np.copy(W_i)

    def get_status(self):

        return {
            'k': self.k,
            'd': self.d,
            'W': self.W,
            'epsilon': self.epsilon,
            'get_optimizer': self.get_optimizer}
