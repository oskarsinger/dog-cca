import numpy as np
import numpy.random as npr
import data.loaders.synthetic.shortcuts as dlss

from runners.bandit import FiniteHyperBandRunner as FHBR
from appgrad import NViewAppGradCCA as NVAGCCA
from arms.appgrad import NViewAppGradCCAArm as NVAGCCAA
from data.servers.minibatch import Minibatch2Minibatch as M2M
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BCGS
from drrobert.arithmetic import int_ceil_log as icl
from drrobert.random import log_uniform as lu
from optimization.optimizers.quasinewton import FullAdamOptimizer as FADO
from optimization.stepsize import InverseSquareRootScheduler as ISRS

def run_n_view_online_appgrad_cosine_data_hyperband_experiment(
    k, n, ps, 
    periods, amplitudes, phases, indexes,
    max_rounds=10):

    max_size = n
    min_size = int(min(periods)) * 3
    dl_list = dlss.get_cosine_loaders(
        ps, periods, amplitudes, phases, indexes,
        period_noise, phase_noise, amplitude_noise)
    # TODO: offer different centering options
    # 1. long term physiological averages
    # 2. moving averages; new hyperparameter for forget factor; ugh
    ds_list = [M2M(loader, bs, center=True) 
               for loader in loaders]
    dimensions = [ds.cols() for ds in ds_list]
    runner = FHBR(
        RandomArmSampler(dimensions, k, k+icl(k)+1).get_arm
        ds_list,
        max_rounds,
        max_size,
        min_size)

    runner.run()

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
