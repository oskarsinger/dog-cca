import numpy as np

from theline.svd import get_svd_power

def get_easy_DCCAPMLs(dynamics, num_data, k, ds, lazy=True):

    Z = np.random.randn(k, num_data)
    Psi_inits = [np.random.randn(d * 2, d)
                 for d in ds]
    Psis = [np.dot(Pi.T, Pi) for Pi in Psi_inits]
    Ws = [np.random.randn(d, k) for d in ds]
    mus = [np.random.randn(d, 1) for d in ds]
    zipped = zip(
        dynamics,
        Ws,
        Psis,
        mus)
    DCCAPML = DynamicCCAProbabilisticModelLoader
    
    return [DCCAPML(d, W, Psi, mu, Z, lazy=lazy)
            for (d, W, Psi, mu) in zipped]

# TODO: see if its okay to use svd and non-symmetric dynamics
class DynamicCCAProbabilisticModelLoader:

    def __init__(self,
        dynamics, 
        W, 
        Psi, 
        mu, 
        Z, 
        lazy=True):
        
        self.dynamics = dynamics
        self.W = W
        self.Psi = Psi
        self.mu = mu
        self.Z = Z
        self.lazy = lazy

        self.d = self.W.shape[0]
        (self.k, self.num_data) = self.Z.shape
        (lam, self.Q) = np.linalg.eig(dynamics)
        self.lam = lam[:,np.newaxis]

        QW = np.dot(self.Q, self.W)

        self.mean = np.dot(QW, Z) + self.mu

        for t in range(Z.shape[0]):
            self.mean[:,t:] *= self.lam

        self.mean = np.dot(self.Q.H, self.mean)
        self.sd = get_svd_power(self.Psi, 0.5)
        self.data = None

        if not self.lazy:
            self._set_data()

    def get_data(self):

        if self.data is None:
            self._set_data()

        return self.data

    def _set_data(self):

        init = np.random.randn(
            self.num_data, self.d)
        scaled = np.dot(init, self.sd)
        
        self.data = scaled + self.mean.T
