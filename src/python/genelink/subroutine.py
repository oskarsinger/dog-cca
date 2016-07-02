import numpy as np

import global_utils as gu

from linal.qr import get_q
from linal.utils import multi_dot, quadratic as quad
from linal.utils import get_mahalanobis_inner_product as get_mip
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

class GenELinKSubroutine:

    def __init__(self,
        k, d,
        epsilon=10**(-3),
        max_iter=100,
        get_optimizer=None,
        verbose=False):

        # Set the easy ones
        self.k = k
        self.d = d
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.verbose = verbose

        # Verify and set the optimizer factory
        if get_optimizer is None:
            get_optimizer = MAG

        self.get_optimizer = get_optimizer

        # Initialize object-wide state variable for W
        self.W = None
        self.num_rounds = 0

    def get_update(self, A, B, eta):

        (nA, pA) = A.shape
        (nB, pB) = B.shape
        ds = [nA, pA, nB, pB]

        if not len(set(ds + [self.d])) == 1:
            raise ValueError(
                'All dimensions of both A and B should be equal.')

        # Initializing optimizer for this round
        optimizer = self.get_optimizer()
        
        # Create inner product for Gram-Schmidt orthogonalization
        inner_prod = get_mip(B)

        if self.W is None:
            unnormed = np.random.randn(self.d, self.k)
            self.W = get_q(unnormed, inner_prod=inner_prod)

        # Initialize iteration variables
        B_term = get_svd_power(quad(self.W, B), power=-1)
        A_term = quad(self.W, A)
        W_i = multi_dot([self.W, B_term, A_term])
        W_i1 = None
        converged = False

        while not converged and self.num_rounds < self.max_iter:

            if self.verbose:
                print 'GenELinKSubroutine Iteration:', self.num_rounds

            # Update iteration variable
            eta_i = eta / (self.num_rounds + 1)**(0.5)

            # Get new parameter estimate
            gradient = np.dot((0.5*B - A).T, W_i)
            W_i1 = optimizer.get_update(W_i, gradient, eta_i)

            if self.verbose:
                print 'GenELinKSubroutine checking for convergence'

            # Check for convergence
            converged = gu.misc.is_converged(
                [(W_i, W_i1)], [self.epsilon], self.verbose)

            # Update iteration variables
            W_i = np.copy(W_i1)
            self.num_rounds += 1

        # Update global state of W with normalized W_i
        self.W = get_q(W_i, inner_prod=inner_prod)

        return (np.copy(self.W), converged)

    def get_status(self):

        return {
            'k': self.k,
            'd': self.d,
            'W': self.W,
            'num_rounds': self.num_rounds,
            'max_iter': self.max_iter,
            'epsilon': self.epsilon,
            'get_optimizer': self.get_optimizer}
