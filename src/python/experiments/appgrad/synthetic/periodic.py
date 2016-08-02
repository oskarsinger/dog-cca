import numpy as np

from data.loaders.synthetic import CosineLoader as CL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl

from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_experiment(
    n, ps, periods, amplitudes, phases, indexes,
    max_iter=10,
    exps=None, windows=None,
    etas=None, lowers=None,
    verbose=False):

    lens = set([
        len(ps),
        len(periods),
        len(amplitudes),
        len(phases),
        len(indexes)])

    if not len(lens) == 1:
        raise ValueError(
            'Args periods, amplitudes, and phases must all have same length.')

    bs = 3
    loader_info = zip(
        ps,
        periods,
        amplitudes,
        phases,
        indexes)
    loaders = [CL(p, max_rounds=n, period=per, amplitude=a, phase=ph, index=i)
               for (p, per, a, ph, i) in loader_info]
    servers = [M2M(loader, bs, center=True) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, 1,
        max_iter=max_iter,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        verbose=verbose)
