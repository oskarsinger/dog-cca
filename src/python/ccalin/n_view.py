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
        gs_list=None):

        if not agu.is_k_valid(ds_list, k):
            raise ValueError(
                'The value of k must be less than or equal to the minimum of the' +
                ' number of columns of X and Y.')

        self.k = k
        self.ds_list = ds_list
        self.num_views = len(self.ds_list)

        if gs_list is None:
            gs_list = [BCGS() for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must have length of ds_list.')

        self.gs_list = gs_list    

        self.num_rounds = 0
        self.has_been_fit = False
        self.Phis = None

        def fit(self, 
            max_iter=10000, 
            eta = 0.1,
            verbose=False):

            subroutine = GLKS(self.k, self.d)

            # Initialize iteration variables
            converged = False
            i = 0

            while not converged and i < max_iter:
                
                (Xs, Sxs) = gu.data.get_batch_and_gram_lists(
                    self.ds_list, self.gs_list)

                A = ccalinu.get_A(Xs)
                B = ccalinu.get_B(Sxs)
                W_i = subroutine.get_update(A, B, eta)

                # TODO: Check for convergence here

                i += 1

            # TODO: Extract Phis from GenELinK subroutine

            self.Phis = "Something"

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
        pre_Wxs = ccalinu.get_pre_Wxs(gep_solution, self.gs_list, self.k)

        # Normalize bases
        self.Phis = ccalinu.get_normed_Wxs(pre_Wxs, Sxs)

        # Update model state
        self.has_been_fit = True

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return [np.copy(Phi) for Phi in self.Phis]
