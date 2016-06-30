from appgrad import AppGradCCA, NViewAppGradCCA
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BGS

def run_online_ccalin_experiment(
    X_server, Y_server, k,
    exp=False, verbose=True,
    eta1=0.1, eta2=0.1, 
    lower1=None, lower2=None):

    model = AppGradCCA(k, online=True)
    (X_gs, Y_gs) = (None, None)

    if exp:
        X_gs = EGS()
        Y_gs = EGS()

    X_optimizer = None
    Y_optimizer = None
    
    if lower1 is not None:
        X_optimizer = MAG(lower=lower1)

    if lower2 is not None:
        Y_optimizer = MAG(lower=lower2)

    model.fit(
        X_server, Y_server, 
        X_gs=X_gs, Y_gs=Y_gs,
        X_optimizer=X_optimizer,
        Y_optimizer=Y_optimizer,
        verbose=verbose)

    return model

def run_online_n_view_ccalin_experiment(
    servers, k, 
    exps=None, windows=None,
    etas=None, lowers=None,
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

    print "Creating model object"
    model = NViewAppGradCCA(
        k, servers, gs_list=gram_servers, online=True)

    optims = None

    print "Creating optimizers"
    if (lowers is not None) and (len(lowers) == len(servers)):
        optims = [MAG(lower=lower)
                  for lower in lowers] + \
                 [MAG()]

    print "Fitting model"
    model.fit(
        optimizers=optims,
        etas=etas,
        verbose=verbose)

    return model
