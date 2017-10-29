import numpy as np

from fitterhappier import LinearConjugateGradientOptimizer as LCGO

class DistributedHComputer:

    def __init__(self, X_server, K=1, G_init=None):

        self.X = self.X_server.get_data()
        self.K = K

        if G_init is None:
            G_init = np.random.randn(
                self.X.shape[0], self.K)

        self.G = G_init
        self.H = None
        
    def get_C(self):

        return get_C_or_H(self.X, self.G)

    def get_v(self, P):

        self.H = get_C_or_H(self.X, P)
        (U, s, VT) = np.thelineg.svd(
            self.H, full_matrices=False)
        self.new_G = np.dot(U, VT)
        GH = np.dot(self.new_G.T, self.H)

        return np.trace(GH)

    def update_G(self):

        self.G = np.copy(self.new_G)

def get_H(Xs, Gs, v):

    Xvs = [X for (w, X) in enumerate(Xs)
           if not w == v]
    Gvs = [G for (w, G) in enumerate(Gs)
           if not w == v]
    Cvs = [get_C_or_H(Xw, Gw)
           for (Xw, Gw) in zip(Xvs, Gvs)]
    Pv = sum(Cvs)

    return get_C_or_H(Xs[v], Pv)

def get_C_or_H(X, G_or_P):

    lcgo = LCGO(X, G_or_P)

    lcgo.run()

    R_or_E = lcgo.get_parameters()

    return np.dot(X, R_or_E)
