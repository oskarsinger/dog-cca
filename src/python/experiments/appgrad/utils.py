from appgrad import AppGradCCA, NViewAppGradCCA
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import ExpGramServer as EGS
from data.servers.gram import BoxcarGramServer as BGS
from data.servers.percentile import Data2Percentiles as D2P

def run_online_n_view_appgrad_experiment(
    servers, k, 
    percentiles=None,
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

    if percentiles is not None:
        servers = [D2P(ds, ps) 
                   for (ds, ps) in zip(servers, percentiles)]

    print "Creating model object"
    model = NViewAppGradCCA(
        k, servers, 
        gs_list=gram_servers, 
        online=True,
        verbose=verbose)

    optims = None

    print "Creating optimizers"
    if (lowers is not None) and (len(lowers) == len(servers)):
        optims = [MAG(lower=lower, verbose=verbose)
                  for lower in lowers]

    print "Fitting model"
    model.fit(
        optimizers=optims,
        etas=etas)

    return model
