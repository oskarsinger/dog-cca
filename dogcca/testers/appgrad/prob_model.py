import numpy as np

from drrobert.misc import unzip
from drrobert.stats import get_zm_uv
from whitehorses.loaders.multiview.cca import get_easy_SCCAPMLs
from whitehorses.servers.minibatch import Batch2Minibatch as B2M
from whitehorses.servers.queue import ExponentialDownWeightedQueue as EDWQ
from dogcca.appgrad import get_ag_views

class CCAProbabilisticModelAppGradTester:

    def __init__(self, ds, k=2, num_data=1000, delay=None):

        self.ds = ds
        self.k = k
        self.num_data = num_data
        self.delay = delay

        self.num_views = len(ds)
        self.loaders = get_easy_SCCAPMLs(
            self.num_data,
            self.k,
            self.ds)
        print(self.loaders[0].get_data().shape)
        bss = [EDWQ(4*k, alpha=0.99) 
               for _ in range(self.num_views)]
        self.servers = [B2M(1, data_loader=dl)
                        for dl in self.loaders]
        self.agvs = get_ag_views(
            self.num_views,
            self.k,
            bss=bss)
        self.Phis = None
        self.tccs = []

    def get_parameters(self):

        return self.Phis

    def run(self):

        printerval = self.num_data / 10

        for t in range(self.num_data):

            zipped = zip(self.agvs, self.servers)

            for (agv, ds) in zipped:
                agv.set_data(ds.get_data())

            if self.delay is None or t % self.delay == 0:
                XPhis = [agv.get_projected()
                         for agv in self.agvs]

                for agv in self.agvs:
                    for (n, XPhi) in enumerate(XPhis):
                        agv.update_neighbor_state(n, XPhi)

            (self.Phis, new_tccs) = unzip(
                [agv.get_update() 
                 for agv in self.agvs])
            self.tccs.append(new_tccs)

            if False: #t % printerval == 0:
                print('tccs', new_tccs)

        zipped = zip(self.Phis, self.loaders)
        XPhis = [np.dot(get_zm_uv(dl.get_data()), Phi)
                 for (Phi, dl) in zipped]
        tcc_mats = [np.dot(XPhis[n].T, XPhis[m])
                    for n in range(self.num_views)
                    for m in range(n, self.num_views)]
        tccs = [np.trace(tccm) / self.k
                for tccm in tcc_mats]
        
        print(tccs)
