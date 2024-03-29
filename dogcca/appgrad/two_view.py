import numpy as np

import utils as agu
import global_utils as gu

from optimization.optimizers import GradientOptimizer as GO
from optimization.utils import get_gram
from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS

class AppGradCCA:

    def __init__(self,
        k=1,
        online=False,
        eps1=10**(-3), eps2=10**(-3)):

        self.k = k
        self.online = online
        self.eps1 = eps1
        self.eps2 = eps2

        self.has_been_fit = False
        self.Phi = None
        self.unn_Phi = None
        self.Psi = None
        self.unn_Psi = None
        self.history = []

    def fit(self, 
        X_ds, Y_ds, 
        X_gs=None, Y_gs=None,
        X_optimizer=None, Y_optimizer=None,
        eta1=0.001, eta2=0.001,
        verbose=False,
        max_iter=10000):

        if X_optimizer is None:
            X_optimizer = GO(verbose=verbose)

        if Y_optimizer is None:
            Y_optimizer = GO(verbose=verbose)

        if X_gs is None:
            X_gs = BCGS() if self.online else BGS()

        if Y_gs is None:
            Y_gs = BCGS() if self.online else BGS()

        (X, Sx, Y, Sy) = self._init_data(X_ds, Y_ds, X_gs, Y_gs)

        print "Getting initial basis estimates"

        # Randomly initialize normalized and unnormalized canonical bases for
        # timesteps t and t+1. Phi corresponds to X, and Psi to Y.
        basis_pairs = agu.get_init_basis_pairs([Sx, Sy], self.k)
        (Phi_t, unn_Phi_t) = basis_pairs[0]
        (Psi_t, unn_Psi_t) = basis_pairs[1]
        (Phi_t1, unn_Phi_t1, Psi_t1, unn_Psi_t1) = (None, None, None, None)

        # Initialize iteration-related variables
        converged = [False] * 2
        i = 1

        while (not all(converged)) and i <= max_iter:

            # Update step scales for gradient updates
            eta1_i = eta1 / i**0.5
            eta2_i = eta2 / i**0.5
            i = i + 1

            if verbose:
                print "Iteration:", i
                print "\teta1:", eta1_i, "\teta2:", eta2_i

                if self.online:
                    print "\tGetting updated minibatches and Sx and Sy"

            # Update random minibatches if doing SGD
            if self.online:
                X = X_ds.get_data()
                Sx = X_gs.get_gram(X)
                Y = Y_ds.get_data()
                Sy = Y_gs.get_gram(Y)

            if verbose:
                print "\tGetting updated basis estimates"

            # Get unconstrained, unnormalized gradients
            (unn_Phi_grad, unn_Psi_grad) = agu.get_gradients(
                [X,Y], [(unn_Phi_t, Phi_t),(unn_Psi_t, Psi_t)])

            # Make updates to basis parameters
            unn_Phi_t1 = X_optimizer.get_update(
                    unn_Phi_t, unn_Phi_grad, eta1_i)
            unn_Psi_t1 = Y_optimizer.get_update(
                    unn_Psi_t, unn_Psi_grad, eta2_i)

            # Normalize updated bases
            Phi_t1 = agu.get_gram_normed(unn_Phi_t1, Sx)
            Psi_t1 = agu.get_gram_normed(unn_Psi_t1, Sy)

            if verbose:
                Phi_dist = np.thelineg.norm(unn_Phi_t1 - unn_Phi_t)
                Psi_dist = np.thelineg.norm(unn_Psi_t1 - unn_Psi_t)
                print "\tDistance between unnormed Phi iterates:", Phi_dist
                print "\tDistance between unnormed Psi iterates:", Psi_dist
                print "\tObjective:", agu.get_objective(
                    [X, Y], [Phi_t1, Psi_t1])

            # Check for convergence
            converged = gu.misc.is_converged(
                [(unn_Phi_t, unn_Phi_t1), (unn_Psi_t, unn_Psi_t1)],
                [self.eps1, self.eps2],
                verbose)

            # Update state
            (unn_Phi_t, Phi_t, unn_Psi_t, Psi_t) = (
                np.copy(unn_Phi_t1),
                np.copy(Phi_t1),
                np.copy(unn_Psi_t1),
                np.copy(Psi_t1))

        print "Completed in", str(i), "iterations."

        self.has_been_fit = True
        self.Phi = Phi_t
        self.unn_Phi = unn_Phi_t
        self.Psi = Psi_t
        self.unn_Psi = unn_Psi_t

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Model has not yet been fit.')

        return (self.Phi, self.Psi, self.unn_Phi, self.unn_Psi)

    def _init_data(self, X_ds, Y_ds, X_gs, Y_gs):

        if not agu.is_k_valid([X_ds, Y_ds], self.k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        X = X_ds.get_data()
        Sx = X_gs.get_gram(X)
        Y = Y_ds.get_data()
        Sy = Y_gs.get_gram(Y)

        if not self.online:
            # Find a better solution to this
            n = min([X.shape[0], Y.shape[0]])
            if X.shape[0] > n:
                # Remove to-be-truncated examples from Gram matrix
                Sx -= get_gram(X[n:,:])

                # Truncate extra examples
                X = X[:n,:]
            elif Y.shap[0] > n:
                # Do the same for Y if Y has the extra examples
                Sy -= get_gram(Y[n:,:])
                Y = Y[:n,:]

        return (X, Sx, Y, Sy)
