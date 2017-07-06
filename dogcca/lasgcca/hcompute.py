import numpy as np

from fitterhappier import LinearConjugateGradientOptimizer as LCGO

class DistributedHComputer:

    def __init__(self, X_server, K=1): 

        self.X = self.X_server.get_data()
        self.K = K
        
    def get_C_or_H(self, G_or_P):

        return get_C_or_H(self.X, G_or_P)

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
