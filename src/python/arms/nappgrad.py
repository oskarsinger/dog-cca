import numpy as np
import drrobert.debug as drdb

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
        self.Xs = [np.zeros((d, self.batch_size))
                   for d in self.dimensions]
        self.Sxs = [np.zeros((d,d))
                    for d in self.dimensions]
        self.missing = None
        self.num_rounds = 0

    def update(self, batches):

        updates = []

        print 'Inside arm.update'

        for batch in batches:

            print '\tInside arm.update loop'

            self.missing = [isinstance(view, MissingData)
                            for view in batch]
            
            print '\t Setting self.Xs'

            self.Xs = [self.Xs[i] \
                        if self.missing[i] else \
                        np.copy(batch[i])
                       for i in xrange(self.num_views)]

            for (i, X) in enumerate(self.Xs):
                drdb.check_for_nan_or_inf(
                    X, 'NVAGCCAA update', 'X_' + str(i))
                           
            get_Sx = lambda i: self.gs_list[i].get_gram(batch[i])

            print '\t Setting self.Sxs'

            self.Sxs = [self.Sxs[i] if self.missing[i] else get_Sx(i)
                        for i in xrange(self.num_views)]

            for (i, Sx) in enumerate(self.Sxs):
                drdb.check_for_nan_or_inf(
                    Sx, 'NVAGCCAA update', 'Sx_' + str(i))
                           
            etas = [es.get_stepsize()
                    for es in self.stepsize_schedulers]
        
            self.num_rounds += 1

            update = self.model.update(
                self.Xs, self.Sxs, self.missing, etas)

            print 'Update has length', len(update)

            updates.append(update)

            print 'Updates has length', len(updates)

        return updates
        
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
