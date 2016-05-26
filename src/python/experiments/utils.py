from appgrad import AppGradCCA, NViewAppGradCCA
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import ExpGramServer as EGS

def run_online_appgrad_experiment(
    X_server, Y_server, k,
    exp=False, verbose=True,
    eta1=0.1, eta2=0.1, lower=None):

    model = AppGradCCA(k, online=True)
    (X_gs, Y_gs) = (None, None)

    if exp:
        X_gs = EGS()
        Y_gs = EGS()

    X_optimizer = None
    Y_optimizer = None
    
    if lower is not None:
        X_optimizer = MAG(lower=lower)
        Y_optimizer = MAG(lower=lower)

    model.fit(
        X_server, Y_server, 
        X_gs=X_gs, Y_gs=Y_gs,
        X_optimizer=X_optimizer,
        Y_optimizer=Y_optimizer,
        verbose=verbose)

    return model

def run_online_n_view_appgrad_experiment(
    servers, k, 
    exp=False, verbose=True,
    etas=None, lower=None):

    model = NViewAppGradCCA(
        k, len(servers), etas=etas, online=True)
    gram_servers = None

    if exp:
        gram_servers = [EGS() for i in range(len(servers)+1)]

    optims = None

    if lower is not None:
        optims = [MAG(lower=lower)
                  for i in range(len(servers))] + \
                 [MAG()]

    model.fit(
        servers, 
        gs_list=gram_servers,
        optimizers=optims,
        verbose=verbose)

    return model
