import numpy as np

def get_H(Xs, Gs, v):

    Xvs = [X for (w, X) in enumerate(Xs)
           if not w == v]
    Gvs = [G for (w, G) in enumerate(Gs)
           if not w == v]
    zipped = zip(Xvs, Gvs)
    enumed = enumerate(zipped)
    Rvs = [get_CG(Xw, Gw)
           for (w, (Xw, Gw)) in enumed]
    Cvs = [np.dot(Xw, Rw)
           for (Xw, Rw) in zip(Xvs, Rvs)]
    Pv = sum(Cvs)
    Ev = get_CG(Xs[v], Pv)

    return np.dot(Xs[v], Ev)

def get_CG(X, G):
    pass
