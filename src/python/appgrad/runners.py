from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS
from optimization.stepsize import FixedScheduler as FXS

import global_utils.server_tools as gust
import global_utils as gu

class NViewAppGradCCARunner:

    def __init__(self, 
        model, ds_list, 
        eta_schedulers=None,
        gs_list=None,
        max_iter=1,
        verbose=False):

        gu.misc.check_k(ds_list, model.get_status()['k'])

        self.model = model
        self.ds_list = ds_list
        self.max_iter = max_iter
        self.num_views = len(self.ds_list)

        if eta_schedulers is None:
            eta_schedulers = [FXS(0.1) 
                              for i in xrange(self.num_views)]
        elif not len(eta_schedulers) == self.num_views:
            raise ValueError(
                'Parameter eta_schedulers must have length of ds_list.')

        self.eta_schedulers = eta_schedulers

        if gs_list is None:
            gs_list = [BCGS() if self.online else BGS()
                       for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must have length of ds_list.')

        self.gs_list = gs_list

        self.num_rounds = 0
        self.num_iters = 0
        self.converged = False

    def run(self):

        while self.num_iters < self.max_iters and not self.converged:

            finished = any(
                [ds.finished for ds in self.ds_list])
            (Xs, Sxs) = [None] * 2

            while not finished:
                etas = [es.get_stepsize(self.num_rounds) 
                        for es in self.eta_schedulers]
                (Xs, Sxs, missing) = gust.get_batch_and_gram_lists(
                    self.ds_list, self.gs_list, Xs, Sxs)

                self.model.update(Xs, Sxs, missing, etas) 

                self.converged = self.model.get_status()['converged']
                self.num_rounds += 1

            self.num_iters += 1

            for ds in ds_list:
                ds.refresh()
                
    def get_status(self):

        return {}
