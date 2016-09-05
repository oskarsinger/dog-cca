import numpy as np

from data.servers.gram import BoxcarGramServer as BCGS, BatchGramServer as BGS
from optimization.stepsize import InverseSquareRootScheduler as ISRS
from data.pseudodata import MissingData
from appgrad import NViewAppGradCCA as NVAGCCA

class NViewAppGradCCAArm:

    def __init__(self,
        model, 
        num_views,
        batch_size,
        dimensions,
        stepsize_schedulers=None,
        gs_list=None,
        verbose=False):

        self.model = model
        self.num_views = num_views
        self.batch_size = batch_size
        self.dimensions = dimensions

        if stepsize_schedulers is None:
            stepsize_schedulers = [ISRS(0.1) 
                              for i in xrange(self.num_views)]
        elif not len(stepsize_schedulers) == self.num_views:
            raise ValueError(
                'Parameter stepsize_schedulers must have length of num_views.')

        self.stepsize_schedulers = stepsize_schedulers

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

    def update(self, batches):

        if self.Xs is None:
            self.Xs = [np.zeros((d, self.batch_size))
                       for d in self.dimensions]

        if self.Sxs is None:
            self.Sxs = [np.zeros((d,d))
                        for d in self.dimensions]

        self.missing = [isinstance(batch, MissingData)
                        for batch in batches]
        self.Xs = [self.Xs[i] \
                    if self.missing[i] else \
                    np.copy(batches[i])
                   for i in xrange(self.num_views)]
        get_Sx = lambda i: self.gs_list[i].get_gram(batches[i])
        self.Sxs = [self.Sxs[i] if self.missing[i] else get_Sx(i)
                    for i in xrange(self.num_views)]
        etas = [es.get_stepsize(self.num_rounds)
                for es in self.stepsize_schedulers]
    
        self.model.update(
            self.Xs, self.Sxs, self.missing, etas)

        self.num_rounds += 1
        
    def get_status(self):

        return {
            'gs_list': self.gs_list,
            'Xs': self.Xs,
            'Sxs': self.Sxs,
            'missing': self.missing,
            'num_rounds': self.num_rounds,
            'batch_size': self.batch_size,
            'model': self.model,
            'stepsize_schedulers': self.stepsize_schedulers}
