import numpy as np

import utils as agu
import global_utils as gu

from drrobert.misc import unzip
from optimization.utils import get_gram
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS

class NViewAppGradCCA:

    def __init__(self,
        k, ds_list, 
        gs_list=None,
        online=False,
        epsilons=None):

        if not gu.misc.is_k_valid(ds_list, k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        self.k = k
        self.ds_list = ds_list
        self.num_views = len(self.ds_list)
        
        if gs_list is None:
            gs_list = [BCGS() if self.online else BGS()
                       for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must have length of ds_list.')

        self.gs_list = gs_list    

        if epsilons is None:
            epsilons = [10**(-4)] * (self.num_views + 1)
        elif not len(epsilons) == self.num_views:
            raise ValueError(
                'Parameter epsilons must have length of ds_list.')

        self.epsilons = epsilons
        self.online = online

        self.num_rounds = 0
        self.has_been_fit = False
        self.basis_pairs = None

        if self.online:
            self.filtering_history = None

    def fit(self,
        optimizers=None,
        etas=None,
        verbose=False,
        max_iter=10000):

        if etas is None:
            etas = [0.00001] * (self.num_views + 1)

        if optimizers is None:
            optimizers = [MAG(verbose=verbose) for i in range(self.num_views + 1)]
        elif not len(optimizers) == self.num_views + 1:
            raise ValueError(
                'Parameter optimizers must be of length num_views+1.')

        print "Getting initial (mini)batches and Gram matrices"

        (Xs, Sxs) = gu.data.init_data(
            self.ds_list, self.gs_list, 
            online=self.online)

        print "Getting initial basis estimates"

        # Initialization of optimization variables
        basis_pairs_t = agu.get_init_basis_pairs(Sxs, self.k)
        basis_pairs_t1 = None

        # Iteration variables
        converged = [False] * self.num_views

        print "Starting optimization"

        while (not all(converged)) and self.num_rounds < max_iter:

            self.num_rounds += 1

            if verbose:
                (unn, normed) = unzip(basis_pairs_t)
                print "\tObjective:", gu.misc.get_objective(Xs, normed)

            # Update step sizes
            etas_i = [eta / self.num_rounds**0.5 for eta in etas]
            
            if verbose:
                print "Iteration:", self.num_rounds
                print "\t".join(["eta" + str(j) + " " + str(eta)
                                 for j, eta in enumerate(etas_i)])
                if self.online:
                    print "\tGetting updated minibatches and grams"

            if self.online:
                # Get new minibatches and Gram matrices
                (Xs, Sxs) = gu.data.get_batch_and_gram_lists(
                    self.ds_list, self.gs_list)

                self._update_filtering_history(Xs, basis_pairs_t)
                
            if verbose:
                print "\tGetting updated basis estimates"

            # Get updated canonical bases
            basis_pairs_t1 = self._get_basis_updates(
                Xs, Sxs, basis_pairs_t, etas_i, optimizers)

            if verbose:
                print "\tGetting updated auxiliary variable estimate"

            # Check for convergence
            pairs = zip(unzip(basis_pairs_t)[0], unzip(basis_pairs_t1)[0])
            converged = gu.misc.is_converged(
                pairs, self.epsilons, verbose) 

            # Update iterates
            basis_pairs_t = [(np.copy(unn_Phi), np.copy(Phi))
                             for unn_Phi, Phi in basis_pairs_t1]

        self.has_been_fit = True
        self.basis_pairs = basis_pairs_t

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Model has not yet been fit.')

        return unzip(self.basis_pairs)[1]

    def _get_basis_updates(self, 
        Xs, Sxs, basis_pairs, etas, optimizers):

        # Get gradients
        gradients = agu.get_gradients(Xs, basis_pairs)

        # Get unnormalized updates
        updated_unn = [optimizers[i].get_update(
                        basis_pairs[i][0], gradients[i], etas[i])
                       for i in range(self.num_views)]

        # Normalize with Gram-parameterized Mahalanobis
        normed_pairs = [(unn, gu.misc.get_gram_normed(unn, Sx))
                        for unn, Sx in zip(updated_unn, Sxs)]

        return normed_pairs

    def _update_filtering_history(self, Xs, basis_pairs):

        if self.filtering_history is None:
            normed = unzip(basis_pairs)[1]
            self.filtering_history = [np.dot(X[-1,:], Phi)
                                      for (X, Phi) in zip(Xs, normed)]
        else:
            normed = unzip(basis_pairs)[1]

            for i in xrange(self.num_views):
                current = self.filtering_history[i]
                new = np.dot(Xs[i][-1,:], normed[i])
                self.filtering_history[i] = np.vstack([current, new])

    def get_status(self):

        return {
            'ds_list': self.ds_list,
            'gs_list': self.gs_list,
            'k': self.k,
            'bases': unzip(self.basis_pairs)[1],
            'num_views': self.num_views,
            'online': self.online,
            'epsilons': self.epsilons,
            'num_rounds': self.num_rounds,
            'has_been_fit': self.has_been_fit,
            'basis_pairs': self.basis_pairs,
            'filtering_history': self.filtering_history}
