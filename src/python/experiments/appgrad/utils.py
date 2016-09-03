from appgrad import AppGradCCA, NViewAppGradCCA
from optimization.optimizers.ftprl import SchattenPCOMIDOptimizer as SPCOMIDO
from optimization.optimizers.ftprl import PeriodicParameterProximalGradientOptimizer as PPPGO
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BGS
from data.servers.masks import PercentileMask as PM

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
        gram_servers = [BGS(window=w) for w in windows]

    if percentiles is not None:
        servers = [PM(ds, ps) 
                   for (ds, ps) in zip(servers, percentiles)]

    print "Creating model object"
    model = NViewAppGradCCA(
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

class RandomArmSampler:

    def __init__(self, 
        dimensions, k, batch_size, 
        verbose=False):

        self.num_views = len(dimensions)
        self.dimensions = dimensions
        self.k = k
        self.batch_size = batch_size

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
        lower = npr.uniform(upper=0.3)

        if np.random_integers(1,2) == 1:
            window = npr.random_integers(1, 100)
            exp = None
        else:
            window = None
            exp = npr.uniform()

        gs_list = None

        if window is None:
            gs_list = [EGS(weight=exp, reg=gram_reg) 
                       for i in xrange(num_views)]
        else:
            gs_list = [BCGS(window=window, reg=gram_reg)
                       for i in xrange(num_views)]

        optimizers = [FADO(
                        delta=delta, 
                        beta1=beta1, 
                        beta2=beta2, 
                        lower=lower)
                      for i in xrange(num_views)]
        stepsize_schedulers = [ISRS(stepsize) 
                               for i in xrange(num_views)]
        model = NVAGCCA(
            self.k,
            optimizers)

        return NVAGCCAA(
            model,
            self.num_views,
            self.batch_size,
            self.dimensions,
            stepsize_schedulers=stepsize_schedulers,
            gs_list=gs_list,
            verbose=verbose)

    def get_status(self):

        return {}
