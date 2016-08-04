import numpy as np
import global_utils as gu

from math import pi
from time import mktime
from datetime import datetime as DT
from bokeh.models import DatetimeTickFormatter
from bokeh.models import DatetimeTicker
from data.pseudodata import MissingData

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

                    new = np.zeros((1, model_info['k']))

                    if not isinstance(Xs[j], MissingData):
                        new = np.dot(Xs[j][-1,:], Phis[j])

                    filtered_Xs[j] = np.vstack([current, new])

    else:
        gss = model_info['gs_list']
        Xs = gu.server_tools.init_data(dss, gss)[0]
        filtered_Xs = [np.dot(X, Phi)
                       for (X, Phi) in zip(Xs, Phis)]

    return filtered_Xs

def set_datetime_xaxis(plot):

    plot.xaxis.ticker = DatetimeTicker()
    plot.xaxis[0].ticker.desired_num_ticks = 28
    plot.xaxis.formatter = DatetimeTickFormatter(
        formats=dict(
            hours=['%T'],
            days=['%d %b'],
            months=['%d %b %T'],
            years=['%d %b %T']))
    plot.xaxis.major_label_orientation = pi/4

def get_loader_names(model_info):

    ds2name = lambda ds: ds.get_status()['data_loader'].name()

    return [ds2name(ds) + ' (' + str(i) + ') '
            for (i, ds) in enumerate(model_info['ds_list'])]

def get_filtering_X_axis(model_info, length, 
    datetime_axis=False, time_scale=None):

    ds = model_info['ds_list'][0]
    dl = ds.get_status()['data_loader']
    dl_info = dl.get_status()
    X_axis = np.arange(length).astype(float)

    if 'seconds' in dl_info:
        X_axis *= dl_info['seconds']

    if datetime_axis:
        start_time = mktime(dl_info['start_times'][0].timetuple())
        X_axis += start_time
        X_axis = [DT.fromtimestamp(ts) for ts in X_axis.tolist()]
    elif time_scale is not None:
        time_scale = 1.0 / float(time_scale)
        X_axis *= time_scale

    return X_axis
