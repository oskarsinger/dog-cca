import numpy as np

from whitehorses.loaders.simple import EmbeddedCosineLoader as ECL

def get_cosine_SCCAPMLs(
    num_data, 
    k, 
    ds, 
    amp=1, 
    period=2*np.pi,
    phase=0,
    v_shift=0,
    lazy=True):

    Z = ECL(
        num_data,
        k,
        phase=phase,
        amplitude=amp,
        period=period).get_data().T
    Psi_inits = [np.random.randn(d * 2, d)
                 for d in ds]
    Psis = [np.dot(Pi.T, Pi) for Pi in Psi_inits]
    Ws = [np.random.randn(d, k) for d in ds]
    mus = [np.random.randn(d, 1) for d in ds]
    zipped = zip(
        Ws,
        Psis,
        mus)
    SCCAPML = StaticCCAProbabilisticModelLoader

    return [SCCAPML(W, Psi, mu, Z, lazy=lazy)
            for (W, Psi, mu) in zipped]

class CosineNoiseCCAProbabilisticModelLoader:

    def __init__(self, W, Psi, mu, Z, lazy=True):
        
        self.W = W
        self.Psi = Psi
        self.mu = mu
        self.Z = Z
        self.lazy = lazy

        self.d = self.W.shape[0]
        (self.k, self.num_data) = self.Z.shape
        self.mean = np.dot(self.W, Z) + self.mu
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
        shifted = init + self.mean.T

        self.data = np.dot(shifted, self.sd)
