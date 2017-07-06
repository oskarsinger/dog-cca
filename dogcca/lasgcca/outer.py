import numpy as np

from .hcompute import get_H
from .hcompute import DistributedHComputer as DHC

class DisGCCA:

    def __init__(self, 
        X_servers, 
        K=1, 
        G_inits=None, 
        epsilon=0.01):
        
        self.X_servers = X_servers
        self.K = K
        self.epsilon = epsilon

        if G_inits is None:
            G_inits = [None] * len(X_servers)

        zipped = zip(
            self.X_servers,
            G_inits)

        self.dhcs = [DHC(X_s, K=self.K, G_init=G)
                     for (X_s, G) in zipped]
        self.objectives = []

    def run(self):

        converged = False

        while not converged: 
            Cs = [dhc.get_C()
                  for dhc in self.dhcs]
            P0 = sum(Cs)
            vs = [dhc.get_v(P0 - Cs[w])
                  for (w, dhc) in enumerate(self.dhcs)]
            w_max = max(
                range(len(vs)), 
                key=lambda w: vs[w])

            self.dhcs[w_max].update_G()
            self.objectives.append(sum(vs))

            current_obj = self.objectives[-1]
            prev_obj = self.objectives[-2]
            diff = current_obj - prev_obj
            ratio = diff / current_obj
            converged = ratio < self.epsilon

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
                (U, s, VT) = np.linalg.svd(
                    H_v, full_matrices=False)
                # TODO: make sure I actually need to transpose V
                new_Gs[v] = np.dot(U, VT)

            dists = [np.linalg.norm(nG - oG)
                     for (nG, pG) in zip(new_Gs, old_Gs)]
            converged = all([d < self.epsilon for d in dists])
            old_Gs = [np.copy(nG) for nG in new_Gs]

        self.Gs = new_Gs


