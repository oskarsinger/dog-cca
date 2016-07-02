import numpy as np

import global_utils as gu
import ccalin.utils as ccalinu

from data.servers.gram import BatchGramServer as BGS
from data.servers.gram import BoxcarGramServer as BCGS
from genelink import BatchGenELinKSolver as BGLKS
from genelink import GenELinKSubroutine as GLKS

class OnlineNViewCCALin:

    def __init__(self,
        k, ds_list,
        gep_solver=None,
        gs_list=None,
        epsilon=10**(-4),
        verbose=False):

        if not gu.misc.is_k_valid(ds_list, k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        self.k = k
        self.ds_list = ds_list
        self.num_views = len(self.ds_list)
        dims = [ds.cols() for ds in self.ds_list]
        self.d = sum(dims)
        self.epsilon = epsilon
        self.verbose = verbose

        if gs_list is None:
            gs_list = [BCGS() for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must have length of ds_list.')

        self.gs_list = gs_list    

        if gep_solver is None:
            gep_solver = GLKS(
                2*self.k, self.d, verbose=self.verbose)

        self.gep_solver = gep_solver

        self.num_rounds = 0
        self.has_been_fit = False
        self.Phis = None

    def fit(self, 
        max_iter=10000, 
        eta = 0.1):

        # Initialize iteration variables
        W_i = None
        W_i1 = None
        converged = [False]

        while not all(converged) and self.num_rounds < max_iter:

            if self.verbose:
                print 'OnlineNViewCCALin Iteration:', self.num_rounds
            
            # Get the new data and gram matrices
            (Xs, Sxs) = gu.data.get_batch_and_gram_lists(
                self.ds_list, self.gs_list)

            # Generate the matrices for the GEP
            A = ccalinu.get_A(Xs)
            B = ccalinu.get_B(Sxs)

            # Initialize GEP convergence variable
            gep_converged = False

            # Get an update from the GenELinK subroutine
            if W_i is None:
                (W_i, gep_converged) = self.gep_solver.get_update(
                    A, B, eta)
            else:
                (W_i1, gep_converged) = self.gep_solver.get_update(
                    A, B, eta)

                if self.verbose:
                    print 'OnlineNViewCCALin checking for convergence'

                # Check for convergence
                converged = gu.misc.is_converged(
                    [(W_i, W_i1)], [self.epsilon], self.verbose)
                W_i = np.copy(W_i1)

            if self.verbose:
                print 'GEP', self.num_rounds, 'converged?:', gep_converged

            self.num_rounds += 1

        # Extract unnormed bases from GEP solution
        pre_Wxs = ccalinu.get_pre_Wxs(W_i, self.ds_list, self.k)

        # Normalize bases
        self.Phis = np.copy(ccalinu.get_normed_Wxs(pre_Wxs, Sxs))

        # Update model state
        self.has_been_fit = True

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return [np.copy(Phi) for Phi in self.Phis]

    def get_status(self):

        return {
            'k': self.k,
            'ds_list': self.ds_list,
            'gs_list': self.gs_list,
            'gep_solver': self.gep_solver,
            'epsilon': self.epsilon,
            'num_views': self.num_views,
            'num_rounds': self.num_rounds,
            'd': self.d,
            'bases': self.Phis,
            'has_been_fit': self.has_been_fit}

class NViewCCALin:

    def __init__(self,
        k, ds_list,
        gs_list=None,
        online=False):

        if not gu.is_k_valid(ds_list, k):
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
        self.online = online

        self.num_rounds = 0
        self.has_been_fit = False
        self.Phis = None

    def fit(self,
        max_iter=10000,
        verbose=False):

        # Get data
        (Xs, Sxs) = gu.data.init_data(ds_list, gs_list)

        # Prepare GEP input
        A = ccalinu.get_A(Xs)
        B = ccalinu.get_B(Sxs)

        # Create GEP solver
        gep_solver = GLK(2*self.k, A, B)

        # Calculate GEP solution
        gep_solver.fit(
            max_iter=max_iter, verbose=verbose)

        # Get GEP solution from solver
        gep_solution = gep_solver.get_basis()

        # Extract unnormed bases from GEP solution
        pre_Wxs = ccalinu.get_pre_Wxs(gep_solution, self.ds_list, self.k)

        # Normalize bases
        self.Phis = np.copy(ccalinu.get_normed_Wxs(pre_Wxs, Sxs))

        # Update model state
        self.has_been_fit = True

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return [np.copy(Phi) for Phi in self.Phis]
