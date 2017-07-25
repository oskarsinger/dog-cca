import numpy as np

from drrobert.stats import get_zm_uv
from linal.utils import get_quadratic, get_multi_dot
from whitehorses.servers.gram import BatchGramServer as SGS

def get_ag_views(num_views, k, bss, pss, ess):

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
            for (idn, (bs, gs, ps, es)) in enumed]

    return agvs

class AppGradView:

    def __init__(self, 
        num_views
        k, 
        idn, 
        batch_server,
        prox_server,
        eta_server):

        self.num_views
        self.k = k
        self.idk = idn
        self.batch_server = batch_server
        self.prox_server = prox_server
        sefl.eta_server = eta_server

        self.neighbor_state = [None] * self.num_views
        self.gram_server = SGS()
        self.unn_Phi = None
        self.Phi = None
        self.d = None
        self.tcc_history = []
        self.num_rounds = 0

    def get_update(self, data):

        if self.num_rounds == 0:
            self.d = data.shape[1]
            self.unn_Phi = np.random.randn(self.d, self.k)

        batch = self.batch_server.get_batch(data)
        neighbor_sum = sum(self.neighbor_state)
        neighbor_term = np.dot(
            batch.T, 
            neighbor_sum)
        gram = self.gram_server.get_update(batch)  
        self_term = self.num_views * np.dot(gram, self.unn_Phi)
        gradient = self_term - neighbor_term
        eta = self.eta_server.get_stepsize()

        self.unn_Phi = self.prox_server.get_update(
            self.unn_Phi, gradient, eta)
        
        pre_sqrt = get_quadratic(self.unn_Phi, gram)
        normalizer = get_svd_power(pre_sqrt, -0.5)
        
        self.Phi = np.dot(self.unn_Phi, normalizer)

        tcc = get_multi_dot([
            self.Phi.T, 
            zm_uv_data.T, 
            neighbor_sum]) / self.k

        self.tcc_history.append(tcc)

        return (self.Phi, tcc)

    def update_neighbor_state(self, idn, Phi):

        self.neighbor_state[idn] = np.copy(Phi)
