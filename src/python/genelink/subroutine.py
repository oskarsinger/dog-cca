import numpy as np

import global_utils as gu

from linal.utils import get_multi_dot, get_quadratic as get_quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import FullAdaGradOptimizer as FAGO

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
            get_optimizer = FAGO

        self.get_optimizer = get_optimizer

        # Initialize object-wide state variable for W
        self.W = None
        self.num_rounds = []
        self.convergence_history = []

    def get_update(self, A, B, eta):

        (nA, pA) = A.shape
        (nB, pB) = B.shape
        ds = [nA, pA, nB, pB]

        if not len(set(ds + [self.d])) == 1:
            raise ValueError(
                'All dimensions of both A and B should be equal.')

        # Initializing optimizer for this round
        optimizer = self.get_optimizer(verbose=self.verbose)
        
        if self.W is None:
            if self.verbose:
                print 'GenELinKSubroutine initializing W'

            unnormed = np.random.randn(self.d, self.k)
            self.W = gu.misc.get_gram_normed(unnormed,B)

        if self.verbose:
            print 'GenELinKSubroutine computing optimization initialization.'

        # Initialize iteration variables
        B_term = get_svd_power(get_quad(self.W, B), power=-1)
        A_term = get_quad(self.W, A)
        W_i = get_multi_dot([self.W, B_term, A_term])
        W_i1 = None
        converged = [False]
        i = 0

        if self.verbose:
            print 'GenELinKSubroutine beginning optimization.'

        while not all(converged) and i < self.max_iter:

            if self.verbose:
                print 'GenELinKSubroutine Iteration:', i

            # Update iteration variable
            eta_i = eta / (i + 1)**(0.5)

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
            i += 1

        self.convergence_history.append(converged)
        self.num_rounds.append(i)

        # Update global state of W with normalized W_i
        self.W = gu.misc.get_gram_normed(W_i, B)

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
