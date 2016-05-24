from appgrad import AppGradCCA, NViewAppGradCCA
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG
from data.servers.gram import ExpGramServer as EGS

def run_online_appgrad_experiment(
    X_server, Y_server, k,
    exp=False, lower=None, verbose=True):

    model = AppGradCCA(k, online=True)
    (X_gs, Y_gs) = (None, None)

    if exp:
        X_gs = EGS()
        Y_gs = EGS()
    
    if lower is None:
        model.fit(
            X_server, Y_server, 
            X_gs=X_gs, Y_gs=Y_gs,
            verbose=verbose)
    else:
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
    exp=False, lower=None, verbose=True):

    model = NViewAppGradCCA(k, len(servers), online=True)
    gram_servers = None

    if exp:
        gram_servers = [EGS() for i in range(len(servers)+1)]

    if lower is None:
        model.fit(
            servers, 
            gs_list=gram_servers,
            verbose=verbose)
    else:
        optims = [MAG(lower=lower)
                  for i in range(len(servers))] + \
                 [MAG()]

        model.fit(
            servers, 
            gs_list=gram_servers,
            optimizers=optims,
            verbose=verbose)

    return model
