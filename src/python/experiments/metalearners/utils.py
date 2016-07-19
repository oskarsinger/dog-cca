from metalearners.periodic import PeriodicMetaLearner
from ccalin import OnlineNViewCCALin
from appgrad import NViewAppGradCCA
from data.servers.gram.online import BoxcarGramServer as BCGS
from data.servers.gram.online import ExpGramServer as EGS

def run_online_n_view_periodic_metalearner_experiment(
    ds_list, k,
    get_learner,
    period_length,
    num_periods,
    get_gs=None,
    max_iter=10,
    verbose=False):

    model = PeriodicMetaLearner(
        k, ds_list,
        get_learner,
        period_length,
        num_periods,
        get_gs=get_gs,
        verbose=verbose)

    model.fit(max_iter=max_iter)

    return model

#TODO: include other args for an n-view AppGrad model
def get_appgrad_factory():

    def get_learner(k, ds_list, gs_list, verbose):

        return NViewAppGradCCA(
            k, ds_list, 
            gs_list=gs_list, 
            online=True,
            verbose=verbose)

    return get_learner

#TODO: include other args for an n-view CCALin model
def get_ccalin_factory():

    def get_learner(k, ds_list, gs_list, verbose):

        return OnlineNViewCCALin(
            k, ds_list, 
            gs_list=gs_list, 
            verbose=verbose)

    return get_learner

def get_boxcar_gs_factory(window):
    
    def get_gs():
        
        return BCGS(window=window)

    return get_gs

def get_exp_gs_factory(weight):
    
    def get_gs():
        
        return EGS(weight=weight)

    return get_gs
