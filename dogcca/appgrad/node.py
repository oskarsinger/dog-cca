import numpy as np

from drrobert.stats import get_zm_uv
from theline.utils import get_quadratic, get_multi_dot
from theline.svd import get_svd_power
from whitehorses.servers.gram import BatchGramServer as BGS
from whitehorses.servers.queue import PlainQueue as PQ
from fitterhappier.stepsize import FixedScheduler as FS
from fitterhappier.qn import DiagonalAdaGradServer as DAGS

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
        gram_reg=10**(-5),
        batch_server=None,
        qn_server=None,
        eta_server=None):

        self.num_views = num_views
        self.k = k
        self.idk = idn

        self.batch_size = 2 * self.k

        # TODO: consider making it bigger than k for conditioning purposes
        if batch_server is None:
            batch_server = PQ(
                self.batch_size, 
                center=True)

        self.batch_server = batch_server

        if qn_server is None:
            qn_server = DAGS()

        self.qns = qn_server

        if eta_server is None:
            eta_server = FS(10**(-3))

        self.eta_server = eta_server

        self.neighbor_state = [None] * self.num_views
        self.unn_Phi = None
        self.X_unn_Phi = None
        self.Phi = None
        self.d = None
        self.data = None
        self.tcc_history = []
        self.num_rounds = 0

    def set_data(self, data):

        self.batch_server.add_data(data)
        self.batch = self.batch_server.get_batch()
        self.batch /= np.sqrt(self.batch_size)

        if self.num_rounds == 0:
            self.d = data.shape[1]
            self.unn_Phi = np.random.randn(self.d, self.k)
            self.X_unn_Phi = np.dot(self.batch, self.unn_Phi)

            self._update_Phi()

    def get_projected(self):

        return np.dot(self.batch, self.Phi)

    def get_update(self):

        self._update_unn_Phi()
        self._update_Phi()
        self._update_TCC()

        self.num_rounds += 1

        return (self.Phi, self.tcc_history[-1])

    def _update_TCC(self):

        self_term = np.dot(self.Phi.T, self.batch.T)
        tcc_matrices = [np.dot(self_term, ns)
                        for ns in self.neighbor_state]
        tccs = [np.trace(tccm) / self.k
                for tccm in tcc_matrices]

        self.tcc_history.append(tccs)

    def _update_unn_Phi(self):

        neighbor_sum = sum(self.neighbor_state)
        neighbor_term = np.dot(
            self.batch.T, 
            neighbor_sum)
        self_term = self.num_views * np.dot(
            self.batch.T, self.X_unn_Phi)
        gradient = self_term - neighbor_term
        qn_transd = self.qns.get_transform(gradient)
        eta = self.eta_server.get_stepsize()

        self.unn_Phi -= eta * qn_transd
        self.X_unn_Phi = np.dot(self.batch, self.unn_Phi)
            
    def _update_Phi(self):

        pre_sqrt = np.dot(self.X_unn_Phi.T, self.X_unn_Phi)
        pre_sqrt_norm = np.thelineg.norm(pre_sqrt)
        normalizer = get_svd_power(pre_sqrt, -0.5)
        
        self.Phi = np.dot(self.unn_Phi, normalizer)

    def update_neighbor_state(self, idn, XPhi):

        self.neighbor_state[idn] = np.copy(XPhi)
