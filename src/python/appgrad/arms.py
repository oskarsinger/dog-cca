from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS
from optimization.stepsize import InverseSquareRootScheduler as ISRS
from data.pseudodata import MissingData

class NViewAppGradCCAArm:

    def __init__(self,
        model, 
        num_views,
        batch_size
        dimensions,
        eta_schedulers=None,
        gs_list=None,
        verbose=False):

        self.model = model
        self.num_views = num_views
        self.batch_size = batch_size
        self.dimensions = dimensions

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
        self.num_rounds = 0

    def update(self, batch_list):

        if self.Xs is None:
            self.Xs = [np.zeros((d, self.batch_size))
                       for d in self.dimensions]

        if self.Sxs is None:
            self.Sxs = [np.zeros((d,d))
                        for d in self.dimensions]

        self.missing = [isinstance(batch, MissingData)
                        for batch in batch_list]
        self.Xs = [self.Xs[i] if self.missing[i] else batch_list[i]
                   for i in xrange(self.num_views)]
        get_gram_update = lambda i: gs_list[i].get_gram(batch_list[i])
        self.Sxs = [self.Sxs[i] if missing[i] else get_gram_update(i)
                    for i in xrange(self.num_views)]
        etas = [es.get_stepsize(self.num_rounds)
                for es in self.eta_schedulers]
    
        self.model.update(self.Xs, self.Sxs, self.missing, etas)

        self.num_rounds += 1
        
    def get_status(self):

        return {}
