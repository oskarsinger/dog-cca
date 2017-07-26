import numpy as np

from drrobert.misc import unzip
from whitehorses.loaders.multiview.cca import get_easy_SCCAPMLs
from whitehorses.servers.minibatch import Batch2Minibatch as B2M
from dogcca.appgrad import get_ag_views

class CCAProbabilisticModelAppGradTester:

    def __init__(self, ds, k=2, num_data=1000):

        self.ds = ds
        self.k = k
        self.num_data = data

        self.num_views = len(ds)
        self.loaders = get_easy_SCCAPMLs(
            self.num_data,
            self.k,
            self.ds)
        self.servers = [B2M(1, data_loader=dl)
                        for dl in self.loaders]

        # TODO: pass in custom pss and ess
        self.agvs = get_ag_views(
            self.num_views,
            self.k)
        self.Phis = None
        self.tccs = []

    def get_parameters(self):

        return self.Phis

    def run(self):

        interval = t / 10

        for t in range(self.num_data):

            zipped = zip(self.agvs, self.servers)

            for (agv, ds) in zipped:
                agv.set_data(ds.get_data())

            XPhis = [agv.get_projected()
                     for agv in self.agvs]

            for agv in self.agvs:
                for (n, XPhi) in enumerate(XPhis):
                    agv.update_neighbor_state(n, XPhi)

            (self.Phis, new_tccs) = unzip(
                [agv.get_update() 
                 for agv in self.agvs])
            self.tccs.append(new_tccs)

            if t % interval == 0:
                print('tccs', new_tccs)
