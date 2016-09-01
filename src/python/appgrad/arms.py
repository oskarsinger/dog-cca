from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS
from optimization.stepsize import InverseSquareRootScheduler as ISRS
from data.pseudodata import MissingData

class NViewAppGradCCAArm:

    def __init__(self,
        model, num_views,
        eta_schedulers=None,
        gs_list=None,
        verbose=False):

        self.model = model

        if eta_schedulers is None:
            eta_schedulers = [ISRS(0.1) 
                              for i in xrange(self.num_views)]
        elif not len(eta_schedulers) == self.num_views:
            raise ValueError(
                'Parameter eta_schedulers must have length of num_views.')

        self.eta_schedulers = eta_schedulers

        if gs_list is None:
            gs_list = [BCGS() if self.online else BGS()
                       for i in range(self.num_views)]
        elif not len(gs_list) == self.num_views:
            raise ValueError(
                'Parameter gs_list must have length of num_views.')

        self.gs_list = gs_list
        self.Xs = None
        self.Sxs = None
        self.missing = None

    def update(self, batch_list):

        self.missing = [isinstance(batch, MissingData)
                        for batch in batch_list]

        if self.Xs is None:
            self.Xs = [None if self.missing[i] else np.zeros_like(b) 
                       for (i, b) in enumerate(batch_list)]

        self.Xs = [self.Xs[i] if self.missing[i] else batch_list[i]
                   for i in xrange(len(batch_list))]
        self.Sxs = 
