from appgrad import AppGradCCA, NViewAppGradCCA

def run_online_appgrad_experiment(
    X_server, Y_server, k, verbose=True):

    model = AppGradCCA(k, online=True)
    
    model.fit(
        X_server, Y_server, 
        verbose=verbose)

    return model

def run_online_n_view_appgrad_experiment(
    servers, k, verbose=True):

    model = NViewAppGradCCA(k, len(servers), online=True)

    model.fit(
        servers, 
        verbose=verbose)

    return model
