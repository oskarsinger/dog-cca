import numpy as np

from fitterhappier.qn import DiagonalAdamServer as DAS
from fitterhappier.stepsize import InversePowerScheduler as IPS
from whitehorses.gram.online import SumGramServer as SGS
from drrobert.arithmetic import get_running_avg as get_ra
from linal.utils import get_quadratic as get_q
from linal.svd_funcs import get_svd_power as get_svdp

class LocalAppGradView:

    def __init__(self,
        id_number,
        d,
        k=1,
        eta0=0.01,
        gram_server=None,
        qn=None):

        self.idn = id_number
        self.d = d
        self.k = k
        self.eta0 = eta0

        if qn is None:
            qn = DAS()

        self.qn = qn

        if gram_server is None:
            gram_server = SGS()

        self.gs = gram_server

        self.unnPhi = np.random.randn((self.d, self.k))
        self.eta_scheduler = IPS(self.eta0, 0.5)
        self.data_mean = np.zeros((1, self.d))
        self.num_rounds = 0

    def get_update(self, X, neighbor_XPhis):

        self.num_rounds += 1
        self.data_mean = get_ra(
            self.data_mean, 
            X, 
            self.num_rounds)

        centered_X = X - self.data_mean
        # TODO: consider some reweighting of terms here
        neighbor_term = np.dot(
            centered_X,
            sum(neighbor_XPhis))
        gram = self.gs.get_gram(centered_X)
        self_term = np.dot(gram, self.unn_Phi)
        gradient = self_term - neighbor_term
        eta = self.eta_scheduler.get_stepsize()

        self.unnPhi = self.qn.get_transform(
            self.unnPhi,
            gradient,
            eta)

        pre_normalizer = get_q(self.unnPhi, gram)
        normalizer = get_svdp(pre_normalizer, -0.5)
        normed = np.dot(self.unnPhi, normalizer)

        return normed
        # Project new estimate
        # Return projected estimate
