import numpy as np

from whitehorses.loaders.multiview.cca import get_easy_SCCAPMLs
from whitehorses.servers.minibatch import Batch2Minibatch as B2M
from dogcca.appgrad import get_ag_views

class CCAProbabilisticModelAppGradTester:

    def __init__(self, num_views, ds, k=2, num_rounds=1000):

        self.num_views = num_views
        self.ds = ds
        self.k = k
        self.num_rounds = num_rounds

        # TODO: fill in args for loaders
        self.loaders = get_easy_SCCAPMLs()
        self.servers = [B2M(1, data_loader=dl)
                        for dl in self.loaders]

        # TODO: pass in custom pss and ess
        self.agvs = get_ag_views(
            self.num_views,
            self.k)
        self.Phis = None
        self.tccs = None

    def run(self):

        for t in range(self.num_rounds):

            zipped = zip(self.agvs, self.servers)

            for (agv, ds) in zipped:
                agv.set_data(ds.get_data())

            XPhis = [agv.get_projected()
                     for agv in self.agvs]

            for agv in self.agvs:
                for (n, XPhi) in enumerate(XPhis):
                    agv.update_neighbor_state(n, XPhi)

            (self.Phis, self.tccs) = unzip(
                [agv.get_update() 
                 for agv in self.agvs])
