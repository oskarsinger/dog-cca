import numpy as np

from data.loaders.synthetic import CosineLoader as CL
from data.servers.minibatch import Minibatch2Minibatch as M2M
from drrobert.arithmetic import int_ceil_log as icl
from .. import utils as eau

def run_n_view_online_appgrad_cosine_data_experiment(
    n, ps, periods, amplitudes, phases, indexes,
    max_iter=1,
    exps=None, windows=None,
    etas=None, lowers=None,
    cs=None,
    period_noise=True,
    phase_noise=True,
    amplitude_noise=True,
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
    loaders = [_get_CL(p, n, per, a, ph, i,
                  period_noise, phase_noise, amplitude_noise)
               for (p, per, a, ph, i) in loader_info]
    servers = [M2M(loader, bs, center=True) for loader in loaders]

    return eau.run_online_n_view_appgrad_experiment(
        servers, 1,
        max_iter=max_iter,
        exps=exps, windows=windows,
        lowers=lowers, etas=etas,
        periods=periods, cs=cs,
        verbose=verbose)

def _get_CL(
    p,
    max_rounds,
    period,
    amplitude,
    phase,
    index,
    period_noise,
    phase_noise,
    amplitude_noise):

    return CL(
        p,
        max_rounds=max_rounds,
        period=period,
        amplitude=amplitude,
        phase=phase,
        index=index,
        period_noise=period_noise,
        phase_noise=phase_noise,
        amplitude_noise=amplitude_noise)
