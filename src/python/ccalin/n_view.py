import numpy as np

from data.servers.gram import BatchGramServer as BGS
from linal.gep.genelink import GenELinK as GLK
from linal.qr import get_q
from linal.utils import multi_dot

class NViewCCALin:

    def __init__(self,
        k, num_views,
        epsilons=None):

        if epsilons is not None:
            if not len(epsilons) == num_views:
                raise ValueError(
                    'Parameter epsilons should have length num_views.')
        else:
            epsilons = [10**(-5)] * num_views

        self.epsilons = epsilons
        self.k = k
        self.num_views = num_views

        self.has_been_fit = False
        self.Phis = None

    def fit(self,
        ds_list,
        gs_list=None,
        max_iter=10000,
        reg=0.1,
        verbose=False):

        if gs_list is None:
            gs_list = [BGS() for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must be of length num_views.')

        (Xs, Sxs) = ccau.init_data(ds_list, gs_list)
        A = self._get_A(Xs)
        B = self._get_B(Sxs)

        gep_solution = GLK().fit(
            A, B, 2*self.k,
            max_iter=max_iter, verbose=verbose)
        pre_Wxs = self._get_pre_Wxs(gep_solution, gs_list)
        self.Phis = self._get_normed_Wxs(pre_Wxs, Sxs)

        self.has_been_fit = True

    def get_bases(self):

        if not self.has_been_fit:
            raise Exception(
                'Has not yet been fit.')

        return [np.copy(Phi) for Phi in self.Phis]

    def _get_normed_Wxs(self, pre_Wxs, Sxs):

        ips = [lambda x1, x2: multi_dot([x1,Sx,x2])
               for Sx in Sxs]
        
        return [get_q(pre_Wx, inner_product=ip)
                for (pre_Wx, ip) in zip(pre_Wxs, ips)]

    def _get_pre_Wxs(self, gep_solution, gs_list):

        col_list = [gs.cols() for gs in gs_list]
        ends = [sum(col_list[:i+1])
                for i in range(len(col_list))]
        boundaries = zip([0]+ends[:-1], ends)
        U = np.random.randn(2*self.k, self.k)

        return [np.dot(gep_solution[begin:end,:], U)
                for (begin,end) in boundaries]

    def _get_A(Xs):

        dims = [X.shape[1] for X in Xs]
        size = sum(dims)
        A = np.zeros((size, size))

        for i in range(len(Xs)):
            i_start = sum(dims[:i])
            i_end = sum(dims[:i+1])
            for j in range(i+1, len(Xs)):
                j_start = sum(dims[:j])
                j_end = sum(dims[:j+1])
                Sxy = np.dot(Xs[i], Xs[j])
                A[i_start:i_end,j_start:j_end] += Sxy
                A[j_start:j_end,i_start:i_end] += Sxy.T

        return np.copy(A)

    def _get_B(self, Sxs):

        dims = [Sx.shape[0] for Sx in Sxs]
        size = sum(dims)
        B = np.zeros((size, size))

        for i in range(len(Sxs)):
            B[sum(dims[:i]):sum(dims[:i+1])] += Sxs[i]

        return np.copy(B)
