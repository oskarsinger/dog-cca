from ccalin import OnlineNViewCCALin, NViewCCALin
from data.servers.gram import BoxcarGramServer as BGS
from data.servers.gram import ExpGramServer as EGS

def run_online_n_view_ccalin_experiment(
    servers, k, 
    max_iter=10000,
    gep_max_iter=100,
    eta=0.1,
    exps=None, windows=None,
    verbose=True):

    gs_list = None

    print 'Creating gram servers'
    if (exps is not None) and (windows is not None):
        raise ValueError(
            'Only one of exp and window can be set to non-None values.')
    elif exps is not None:
        gs_list = [EGS(weight=w) for w in exps]
    elif windows is not None:
        gs_list = [BGS(window=w) for w in windows]

    print 'Creating model object'
    model = OnlineNViewCCALin(
        k, servers, 
        gs_list=gs_list, 
        max_iter=max_iter,
        gep_max_iter=gep_max_iter,
        verbose=verbose)

    print 'Fitting model'
    model.fit(
        eta=eta)

    return model

def run_batch_n_view_ccalin_experiment(
    servers, k,
    gep_max_iter=100,
    subroutine_max_iter=1000,
    eta=0.1,
    verbose=True):

    print 'Creating model object'
    model = NViewCCALin(
        k, servers,
        gep_max_iter=gep_max_iter,
        subroutine_max_iter=subroutine_max_iter,
        verbose=verbose)

    print 'Fitting model'
    model.fit(eta=eta)

    return model
