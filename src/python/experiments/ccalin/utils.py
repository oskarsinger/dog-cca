from ccalin import OnlineNViewCCALin
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BGS

def run_online_n_view_ccalin_experiment(
    servers, k, 
    max_iter=10000,
    eta=0.1,
    exps=None, windows=None,
    verbose=True):

    gs_list = None

    print "Creating gram servers"
    if (exps is not None) and (windows is not None):
        raise ValueError(
            'Only one of exp and window can be set to non-None values.')
    elif exps is not None:
        gs_list = [EGS(weight=w) for w in exps]
    elif windows is not None:
        gs_list = [BGS(window=w) for w in windows]

    print "Creating model object"
    model = OnlineNViewCCALin(
        k, servers, gs_list=gs_list, verbose=verbose)

    print "Fitting model"
    model.fit(
        max_iter=max_iter,
        eta=eta)

    return model
