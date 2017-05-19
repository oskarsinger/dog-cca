import numpy as np

from pathos.pools import ThreadPool as Pool

class DecentralizedOnlineGraphCCA:

    def __init__(self,
        adj_lists,
        get_model,
        data_servers,
        observable_server,
        neighbor_servers,
        num_rounds=100,
        init_params=None,
        num_threads=24,
        compute_regret=True):

        self.adj_lists = adj_lists
        self.get_model = get_model
        self.data_servers = data_servers
        self.observable_server = observabe_server
        self.neighbor_servers = neighbor_servers
        self.num_nodes = len(self.data_servers)
        self.num_rounds = num_rounds
        self.num_threads = num_threads
        self.compute_regret = compute_regret

        node_info = enumerate(zip(
            self.data_servers,
            self.neighbor_servers))

        self.nodes = [DecentralizedOnlineGraphCCANode(
                        self.get_model(n)
                        ds,
                        ns,
                        n,
                        compute_regret=self.compute_regret)
                      for (n, ds, ns) in node_info]

        for node in self.nodes:
            neighbors = {m: self.nodes[m]
                         for m in self.adj_lists[n]}

            node.set_neighbors(neighbors)

        self.p = self.nodes[0].model.p
        self.np = self.num_nodes * self.p

    def get_parameters(self):

        if self.w_hat is None:
            raise Exception(
                'Parameters have not been computed.')
        
        return self.w_hat

    def compute_parameters(self):

        for t in range(self.num_rounds):
            observables = self.observable_server.get_subset()
            obs_nodes = {n : self.nodes[n]
                         for n in observables}

            #TODO: change to multithreaded once we know it works on single thread
            for node in self.nodes:
                node.communicate()

            for node in obs_nodes:
                node.compute_local_update()

            if self.compute_regret:
                for n in range(self.num_nodes):
                    node = self.nodes[n]
                    regrets_n = node.regrets
                    regret = np.nan

                    if len(regrets_n) > 0:
                        regret = regrets_n[-1]

                    self.regrets[n].append(regret)

class DecentralizedOnlineGraphCCANode:

    def __init__(self,
        model,
        data_server,
        neighbor_server,
        id_number,
        compute_regret=True):

        self.model = model
        self.ds = data_server
        self.ns = neighbor_server
        self.idn = id_number
        self.compute_regret = compute_regret

        self.num_rounds = 0
        self.Phi = None
        self.min_losses = []
        self.regrets = []

    def set_neighbors(self, neighbors):

        self.neighbors = neighbors
        self.nPhis = [None] * len(self.neighbors)

    def communicate(self):

        obs_neighbs = self.ns.get_subset()

        for n in obs_neighbs:
            self.nPhis = [self.neighbors[n].Phi
                          for n in obs_neighbs]
        
    def compute_local_update(self):

        datum = self.ds.get_data()
        
        self.Phi = self.model.get_parameters(
            datum, 
            self.Phi,
            self.nPhis)
