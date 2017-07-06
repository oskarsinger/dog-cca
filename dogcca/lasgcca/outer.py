import numpy as np

from .hcompute import get_H
from .hcompute import DistributedHComputer

class DisGCCA:

    def __init__(self):
        pass

class LasGCCA:

    def __init__(self, 
        Xs, 
        k=1, 
        Gs=None, 
        epsilon=10**(-4)):
        
        self.Xs = Xs
        self.k = k
        self.epsilon = epsilon

        self.num_views = len(self.Xs)
        self.N = self.Xs[0].shape[0]

        if Gs in None:
            Gs = [np.random.randn(self.N, self.k)
                  for _ in range(self.num_views)]

        self.Gs = Gs

    def get_Gs(self):

        return self.Gs

    def run(self):

        mus = [np.mean(X, axis=0)[np.newaxis,:]
               for X in self.Xs]
        centered = [X - mu
                    for (X, mu) in zip(self.Xs, mus)]
        scale = (self.N)**(-0.5)
        scaled = [X * scale for X in centered]
        old_Gs = [np.copy(G) for G in self.Gs]
        new_Gs = [np.copy(oG) for oG in old_Gs]
        converged = False

        while not converged:

            for v in range(self.num_views):
                H_v = get_H(Xs, new_Gs, v)
                (U, s, V) = np.linalg.svd(
                    H_v, full_matrices=False)
                # TODO: make sure I actually need to transpose V
                new_Gs[v] = np.dot(U, V.T)

            dists = [np.linalg.norm(nG - oG)
                     for (nG, pG) in zip(new_Gs, old_Gs)]
            converged = all([d < self.epsilon for d in dists])
            old_Gs = [np.copy(nG) for nG in new_Gs]

        self.Gs = new_Gs


