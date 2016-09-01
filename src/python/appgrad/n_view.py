import numpy as np

import utils as agu
import global_utils as gu
import global_utils.server_tools as gust

from drrobert.misc import unzip
from optimization.utils import get_gram
from optimization.optimizers import GradientOptimizer as GO
from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS

class NViewAppGradCCA:

    def __init__(self,
        k, ds_list, 
        gs_list=None,
        optimizers=None,
        online=False,
        keep_basis_history=False,
        verbose=False,
        epsilons=None):

        gu.misc.check_k(ds_list, k)

        self.k = k
        self.ds_list = ds_list
        self.online = online
        self.keep_basis_history = keep_basis_history
        self.num_views = len(self.ds_list)
        self.verbose = verbose
        
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

        if optimizers is None:
            optimizers = [GO() for i in range(self.num_views)]
        elif not len(optimizers) == self.num_views:
            raise ValueError(
                'Parameter optimizers must be of length num_views.')

        self.optimizers = optimizers

        self.num_rounds = 0
        self.has_been_fit = False
        self.basis_pairs_t = None
        self.basis_pairs_t1 = None
        self.loss_history = []

        if self.online:
            self.filtering_history = None
            self.basis_history = None

    def get_update(self, Xs, Sxs, missing, etas):

        if self.basis_pairs_t is None:
            # Initialization of optimization variables
            self.basis_pairs_t = agu.get_init_basis_pairs(Sxs, self.k)

        self.num_rounds += 1

        if self.verbose:
            (unn, normed) = unzip(basis_pairs_t)
            loss = gu.misc.get_objective(Xs, normed)

            self.loss_history.append(loss)

            print "\tObjective:", loss

        if self.online:
            self._update_history(Xs, missing)
            
        if self.verbose:
            print "\tGetting updated basis estimates"

        # Get updated canonical bases
        self.basis_pairs_t1 = self._get_basis_updates(
            Xs, Sxs, missing, etas)

        if self.verbose:
            print "\tGetting updated auxiliary variable estimate"

        # Check for convergence
        pairs = zip(unzip(basis_pairs_t)[0], unzip(basis_pairs_t1)[0])
        pre_converged = gu.misc.is_converged(
            pairs, self.epsilons, self.verbose)

        # This is because bases are unchanged for missing data and will of course result in zero difference between iterations
        self.converged = [False if missing[i] else c 
                          for (i, c) in enumerate(converged)]

        # Update iterates
        self.basis_pairs_t = [(np.copy(unn_Phi), np.copy(Phi))
                              for unn_Phi, Phi in basis_pairs_t1]

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Model has not yet been fit.')

        return unzip(self.basis_pairs_t)[1]

    def _get_basis_updates(self, Xs, Sxs, missing, etas):

        # Get gradients
        gradients = agu.get_gradients(Xs, basis_pairs)

        # Get basis update for i-th basis 
        get_new_b = lambda i: self.optimizers[i].get_update(
            self.basis_pairs_t[i][0], gradients[i], etas[i])

        # Get unnormalized updates
        updated_unn = [self.basis_pairs_t[i][0] if missing[i] else get_new_b(i)
                       for i in range(self.num_views)]

        # TODO: rewrite this to avoid redundant computation on unchanged bases
        # Normalize with Gram-parameterized Mahalanobis
        normed_pairs = [(unn, gu.misc.get_gram_normed(unn, Sx))
                        for unn, Sx in zip(updated_unn, Sxs)]

        return normed_pairs

    def _update_history(self, Xs, missing):

        normed = unzip(self.basis_pairs_t)[1]

        # TODO: consider storing basis history as list of 3 mode tensors

        if self.filtering_history is None:
            self.filtering_history = [np.dot(X[-1,:], Phi)
                                      for (X, Phi) in zip(Xs, normed)]
            if self.keep_basis_history:
                self.basis_history = [[Phi[:,i][:,np.newaxis].T 
                                       for i in xrange(self.k)]
                                      for Phi in normed]
        else:
            for i in xrange(self.num_views):
                current = self.filtering_history[i]

                # TODO: put meaningful filler here; zeroes are not meaningful
                new = np.zeros((1, self.k))

                if not missing[i]:
                    new = np.dot(Xs[i][-1,:], normed[i])

                self.filtering_history[i] = np.vstack([current, new])

                # Update basis history
                if self.keep_basis_history:
                    for j in xrange(self.k):
                        current = self.basis_history[i][j]
                        new = np.copy(current[-1,:])

                        if not missing[i]:
                            new = normed[i][:,j]

                        self.basis_history[i][j] = np.vstack([current, new])

    def get_status(self):

        return {
            'k': self.k,
            'bases': unzip(self.basis_pairs_t)[1],
            'num_views': self.num_views,
            'online': self.online,
            'epsilons': self.epsilons,
            'num_rounds': self.num_rounds,
            'num_iters': self.num_iters,
            'has_been_fit': self.has_been_fit,
            'basis_pairs': self.basis_pairs,
            'basis_history': self.basis_history,
            'loss_history': self.loss_history,
            'filtering_history': self.filtering_history}
