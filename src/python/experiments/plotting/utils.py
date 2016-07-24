import numpy as np
import global_utils as gu

from time import mktime

def get_refiltered_Xs(model_info):

    dss = model_info['ds_list']
    num_rounds = model_info['num_rounds']
    num_views = model_info['num_views']
    Phis = model_info['bases']
    filtered_Xs = None

    if num_rounds > 1:
        for ds in dss:
            ds.refresh()
    
        for i in range(num_rounds):
            Xs = [ds.get_data() for ds in dss]

            if filtered_Xs is None:
                filtered_Xs = [np.dot(X[-1,:], Phi)
                               for (X, Phi) in zip(Xs, Phis)]
            else:
                for j in xrange(num_views):
                    current = filtered_Xs[j]
                    new = np.dot(Xs[j][-1,:], Phis[j])
                    filtered_Xs[j] = np.vstack([current, new])

    else:
        gss = model_info['gs_list']
        Xs = gu.server_tools.init_data(dss, gss)[0]
        filtered_Xs = [np.dot(X, Phi)
                       for (X, Phi) in zip(Xs, Phis)]

    return filtered_Xs

def get_filtering_X_axis(model_info, length, absolute_time=False, time_scale=None):

    ds = model_info['ds_list'][0]
    dl = ds.get_status()['data_loader']
    dl_info = dl.get_status()
    X_axis = np.arange(length).astype(float)

    if 'seconds' in dl_info:
        X_axis *= dl_info['seconds']

    if absolute_time:
        start_time = mktime(dl_info['start_times'][0].timetuple())
        X_axis += start_time
    elif time_scale is not None:
        time_scale = 1.0 / float(time_scale)
        X_axis *= time_scale

    return X_axis
