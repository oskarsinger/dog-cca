import numpy as np
import numpy.random as npr
import drrobert.debug as drdb

from appgrad import NViewAppGradCCA as NVAGCCA
from arms.nappgrad import NViewAppGradCCAArm as NVAGCCAA
from optimization.optimizers.ftprl import SchattenPCOMIDOptimizer as SPCOMIDO
from optimization.optimizers.ftprl import PeriodicParameterProximalGradientOptimizer as PPPGO
from optimization.optimizers.quasinewton import DiagonalAdamOptimizer as DADO
from optimization.stepsize import FixedScheduler as FXS
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BCGS
from data.servers.masks import PercentileMask as PM
from data.pseudodata import MissingData
from drrobert.random import log_uniform as lu

def run_online_n_view_appgrad_experiment(
    servers, k, 
    max_iter=10,
    percentiles=None,
    exps=None, windows=None,
    etas=None, lowers=None,
    periods=None, cs=None,
    keep_basis_history=False,
    verbose=True):

    gram_servers = None

    print "Creating gram servers"
    if (exps is not None) and (windows is not None):
        raise ValueError(
            'Only one of exp and window can be set to non-None values.')
    elif exps is not None:
        gram_servers = [EGS(weight=w) for w in exps]
    elif windows is not None:
        gram_servers = [BCGS(window=w) for w in windows]

    if percentiles is not None:
        servers = [PM(ds, ps) 
                   for (ds, ps) in zip(servers, percentiles)]

    print "Creating model object"
    model = NVAGCCA(
        k, servers, 
        gs_list=gram_servers, 
        online=True,
        keep_basis_history=keep_basis_history,
        verbose=verbose)

    optims = None

    print "Creating optimizers"
    if cs is not None:
        if lowers is None:
            lowers = [None] * len(servers)

        optims = [PPPGO(period, c, lower=l, verbose=verbose)
                  for (period, c, l) in zip(periods, cs, lowers)]
    elif (lowers is not None) and (len(lowers) == len(servers)):
        optims = [SPCOMIDO(lower=lower, verbose=verbose)
                  for lower in lowers]


    print "Fitting model"
    model.fit(
        max_iter=max_iter,
        optimizers=optims,
        etas=etas)

    return model

class MultiViewDataServer:

    def __init__(self, servers, num_batches=1):

        self.servers = servers
        self.num_batches = num_batches

        print 'Inside MVDS constructor with num_batches', self.num_batches

    def get_data(self):

        batches = []

        for i in xrange(self.num_batches):

            batch = [np.copy(ds.get_data()) 
                     for ds in self.servers]
            
            for view in batch:
                if not isinstance(view, MissingData):
                    try:
                        drdb.check_for_nan_or_inf(
                            view, 'MVDS get_data', 'view_' + str(i))
                    except TypeError:
                        raise Exception(
                            'View was of type' + str(type(view)))

            batches.append(batch)

        #print 'Inside MVDS.get_Data with batches length', len(batches)

        return batches

    def refresh(self):

        for ds in self.servers:
            ds.refresh()

    def get_status(self):

        return {
            'servers': self.servers}

class RandomArmSampler:

    def __init__(self, 
        dimensions, k, batch_size, 
        verbose=False):

        self.num_views = len(dimensions)
        self.dimensions = dimensions
        self.k = k
        self.batch_size = batch_size
        self.verbose = verbose

    def get_arm(self):

        # TODO: make sure I sample all the right parameters
        beta1 = None
        beta2 = None
        stepsize = None
        window = None
        exp = None
        gram_reg = None
        delta = None
        lower = None
        
        (beta1, beta2) = list(npr.uniform(size=2))
        stepsize = lu(10**(-5), 10**(5))
        gram_reg = lu(10**(-5), 10**(5))
        delta = lu(10**(-5), 10**(5))
        lower = npr.uniform(high=0.3)

        if npr.random_integers(1,2) == 1:
            window = npr.random_integers(1, 100)
            exp = None
        else:
            window = None
            exp = npr.uniform()

        parameters = {
            'beta1': beta1,
            'beta2': beta2,
            'stepsize': stepsize,
            'window': window,
            'exp': exp,
            'gram_reg': gram_reg,
            'delta': delta,
            'lower': lower}

        gs_list = None

        if window is None:
            gs_list = [EGS(weight=exp, reg=gram_reg) 
                       for i in xrange(self.num_views)]
        else:
            gs_list = [BCGS(window=window, reg=gram_reg)
                       for i in xrange(self.num_views)]

        optimizers = [DADO(
                        delta=delta, 
                        beta1=beta1, 
                        beta2=beta2, 
                        lower=lower,
                        dual_avg=False)
                      for i in xrange(self.num_views)]
        stepsize_schedulers = [FXS(stepsize) 
                               for i in xrange(self.num_views)]
        model = NVAGCCA(
            self.k,
            optimizers)

        arm = NVAGCCAA(
            model,
            self.num_views,
            self.batch_size,
            self.dimensions,
            stepsize_schedulers=stepsize_schedulers,
            gs_list=gs_list,
            verbose=self.verbose)

        return (arm, parameters)

    def get_status(self):

        return {
            'dimensions': self.dimensions,
            'num_views': self.num_views,
            'batch_size': self.batch_size,
            'k': self.k}
