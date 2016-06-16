import numpy as np
import utils as agu

from drrobert.misc import unzip
from optimization.utils import get_gram
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS

class NViewAppGradCCA:

    def __init__(self,
        k, num_views,
        online=False,
        epsilons=None):

        self.k = k

        if num_views < 2:
            raise ValueError(
                'You must provide at least 2 data servers.')
        else:
            self.num_views = num_views

        if epsilons is not None:
            self.epsilons = epsilons
        else:
            self.epsilons = [10**(-4)] * (self.num_views + 1)

        self.online = online

        self.has_been_fit = False
        self.basis_pairs = None
        self.Psi = None
        self.history = None

    def get_status(self):

        return {
            'k': self.k,
            'num_views': self.num_views,
            'online': self.online,
            'epsilons': self.epsilons,
            'has_been_fit': self.has_been_fit,
            'basis_pairs': self.basis_pairs,
            'Psi': self.Psi,
            'history': self.history}

    def fit(self,
        ds_list, 
        gs_list=None,
        optimizers=None,
        etas=None,
        verbose=False,
        max_iter=100000):

        self.history = []

        if etas is None:
            etas = [0.00001] * (self.num_views + 1)

        if gs_list is None:
            gs_list = [BCGS() if self.online else BGS()
                       for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must be of length num_views.')
            
        if optimizers is None:
            optimizers = [MAG() for i in range(self.num_views + 1)]
        elif not len(optimizers) == self.num_views + 1:
            raise ValueError(
                'Parameter optimizers must be of length num_views+1.')

        (Xs, Sxs) = self._init_data(ds_list, gs_list)

        print "Getting intial basis estimates"

        # Initialization of optimization variables
        basis_pairs_t = agu.get_init_basis_pairs(Sxs, self.k)
        basis_pairs_t1 = None
        bs = ds_list[0].get_status()['batch_size']
        Psi = np.random.randn(bs, self.k)

        # Iteration variables
        converged = [False] * self.num_views
        i = 1

        print "Starting optimization"

        while (not all(converged)) and i < max_iter:

            if verbose:
                (unn, normed) = unzip(basis_pairs_t)
                print "\tObjective:", agu.get_objective(Xs, normed, Psi)

            self.history.append({})

            # Update step sizes
            etas_i = [eta / i**0.5 for eta in etas]
            
            self.history[-1]['etas'] = list(etas_i)

            if verbose:
                print "Iteration:", i
                print "\t".join(["eta" + str(j) + " " + str(eta)
                                 for j, eta in enumerate(etas_i)])
                if self.online:
                    print "\tGetting updated minibatches and grams"

            if self.online:
                # Get new minibatches and Gram matrices
                (Xs, Sxs) = self._get_batch_and_gram_lists(ds_list, gs_list)

            if verbose:
                print "\tGetting updated basis estimates"

            # Get updated canonical bases
            basis_pairs_t1 = self._get_basis_updates(
                Xs, Sxs, basis_pairs_t, Psi, etas_i, optimizers)

            if verbose:
                print "\tGetting updated auxiliary variable estimate"

            # Get updated auxiliary variable
            Psi = self._get_Psi_update(
                Xs, basis_pairs_t1, Psi, etas_i[-1], optimizers[-1])

            # Check for convergence
            pairs = zip(unzip(basis_pairs_t)[0], unzip(basis_pairs_t1)[0])
            converged = agu.is_converged(pairs, self.epsilons, verbose) 

            #self.history[-1]['distances'] = list(distances)
                            
            # Update iterates
            basis_pairs_t = [(np.copy(unn_Phi), np.copy(Phi))
                             for unn_Phi, Phi in basis_pairs_t1]

            i += 1

        self.has_been_fit = True
        self.basis_pairs = basis_pairs_t
        self.Psi = Psi

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Model has not yet been fit.')

        return (self.basis_pairs, self.Psi)

    def _get_basis_updates(self, 
        Xs, Sxs, basis_pairs, Psi, etas, optimizers):

        # Get gradients
        gradients = [agu.get_gradient(Xs[i], basis_pairs[i][0], Psi)
                     for i in range(self.num_views)]

        # Get unnormalized updates
        updated_unn = [optimizers[i].get_update(
                        basis_pairs[i][0], gradients[i], etas[i])
                       for i in range(self.num_views)]

        # Normalize with gram-parameterized Mahalanobis
        normed_pairs = [(unn, agu.get_gram_normed(unn, Sx))
                        for unn, Sx in zip(updated_unn, Sxs)]

        return normed_pairs

    def _get_Psi_update(self, Xs, basis_pairs, Psi, eta, optimizer):

        Phis = [pair[1] for pair in basis_pairs]
        gradient = self._get_Psi_gradient(Psi, Xs, Phis)

        return optimizer.get_update(Psi, gradient, eta)

    def _get_batch_and_gram_lists(self, ds_list, gs_list):

        batch_list = [ds.get_data()
                      for ds in ds_list]
        gram_list = [gs.get_gram(batch)
                     for (gs, batch) in zip(gs_list, batch_list)]

        return (batch_list, gram_list)

    def _get_Psi_gradient(self, Psi, Xs, Phis):

        diffs = [np.dot(X, Phi) - Psi
                 for (X, Phi) in zip(Xs, Phis)]
        residuals = [np.linalg.norm(d) for d in diffs]
        
        self.history[-1]['residuals'] = list(residuals)

        return (2.0 / Psi.shape[0]) * sum(diffs)

    def _init_data(self, ds_list, gs_list):

        if not len(ds_list) == self.num_views:
            raise ValueError(
                'Parameter ds_list must have length num_views.')

        if not agu.is_k_valid(ds_list, self.k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        (Xs, Sxs) = self._get_batch_and_gram_lists(ds_list, gs_list)

        if not self.online:
            # Find a better solution to this
            n = min([X.shape[0] for X in Xs])

            # Remove to-be-truncated examples from Gram matrices
            removed = [X[n:] if X.ndim == 1 else X[n:,:] for X in Xs]
            Sxs = [Sx - get_gram(r) for r, Sx in zip(removed, Sxs)]

            # Truncate extra examples
            Xs = [X[:n] if X.ndim == 1 else X[:n,:] for X in Xs]

        return (Xs, Sxs)
