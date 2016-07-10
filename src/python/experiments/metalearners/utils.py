from metalearners import PeriodicMetaLearner

def run_online_n_view_periodic_metalearner_experiment(
    servers, k,
    get_learner,
    period_length,
    num_periods,
    get_gs=None,
    verbose=False):

    model = PeriodicMetaLearner(
        k, servers,
        get_learner,
        period_length,
        num_periods,
        get_gs=get_gs,
        verbose=verbose)

    model.fit()

    return model

#TODO: include other args for an n-view appgrad model
def get_appgrad_factory():

    def get_learner(k, ds_list, gs_list, verbose):

        return NViewAppGradCCA(
            k, ds_list, 
            gs_list=gs_list, 
            online=True,
            verbose=verbose)

    return get_learner
