import numpy as np

from drrobert.stats import get_zm_uv
from linal.utils import get_quadratic, get_multi_dot
from whitehorses.servers.gram import BatchGramServer as BGS
from whitehorses.servers.queue import PlainQueue as PQ
from fitterhappier.stepsize import FixedScheduler as FS

def get_ag_views(num_views, k, bss=None, pss=None, ess=None):

    if bss is None:
        bss = [None] * num_views

    if pss is None:
        pss = [None] * num_views

    if ess is None:
        ess = [None] * num_views

    zipped = zip(
        bss,
        pss,
        ess)
    enumed = enumerate(zipped)
    agvs = [AppGradView(
                num_views, 
                k, 
                idn, 
                bs,
                ps,
                es)
            for (idn, (bs, ps, es)) in enumed]

    return agvs

class AppGradView:

    def __init__(self, 
        num_views,
        k, 
        idn, 
        batch_server=None,
        prox_server=None,
        eta_server=None):

        self.num_views
        self.k = k
        self.idk = idn

        # TODO: consider making it bigger than k for conditioning purposes
        if batch_server is None:
            batch_server = PQ(k)

        self.batch_server = batch_server

        if prox_server is None:
            prox_server = DAS()

        self.prox_server = prox_server

        if eta_server is None:
            eta_server = FS()

        sefl.eta_server = eta_server

        self.neighbor_state = [None] * self.num_views
        self.gram_server = BGS()
        self.unn_Phi = None
        self.Phi = None
        self.d = None
        self.data = None
        self.tcc_history = []
        self.num_rounds = 0

    def set_data(self, data):

        self.batch_server.add_data(data)
        self.batch = self.batch_server.get_batch()
        self.gram = self.gram_server.get_gram(self.batch)

        if self.num_rounds == 0:
            self.d = data.shape[1]
            self.unn_Phi = np.random.randn(self.d, self.k)

            self._update_unn_Phi()

    def get_projected(self):

        return np.dot(self.data, self.Phi)

    def get_update(self):

        self._update_unn_Phi()
        self._update_Phi()
        self._update_TCC()

        self.num_rounds += 1

        return (self.Phi, self.tcc_history[-1])

    def _update_TCC(self):

        neighbor_sum = sum(self.neighbor_states)
        tcc = get_multi_dot([
            self.Phi.T, 
            self.batch.T, 
            neighbor_sum]) / self.k

        self.tcc_history.append(tcc)

    def _update_unn_Phi(self):

        neighbor_sum = sum(self.neighbor_state)
        neighbor_term = np.dot(
            self.batch.T, 
            neighbor_sum)
        self_term = self.num_views * np.dot(
            self.gram, self.unn_Phi)
        gradient = self_term - neighbor_term
        eta = self.eta_server.get_stepsize()

        self.unn_Phi = self.prox_server.get_update(
            self.unn_Phi, gradient, eta)
            
    def _update_Phi(self):

        pre_sqrt = get_quadratic(self.unn_Phi, self.gram)
        normalizer = get_svd_power(pre_sqrt, -0.5)
        
        self.Phi = np.dot(self.unn_Phi, normalizer)

    def update_neighbor_state(self, idn, XPhi):

        self.neighbor_state[idn] = np.copy(XPhi)
