import global_utils as gu

from data.servers.masks import PeriodicMask as PM
from data.servers.gram.online import BoxcarGramServer as BCGS

class PeriodicMetaLearner:

    def __init__(self, 
        k, ds_list,
        get_learner,
        period_length, 
        num_periods,
        get_gs=None,
        verbose=False):

        gu.misc.check_k(ds_list, k)

        self.k = k
        self.ds_list = ds_list
        self.get_learner = get_learner
        self.pl = period_length
        self.num_periods = num_periods
        self.verbose = verbose

        if get_gs is None:
            get_gs = BCGS

        self.get_gs = get_gs

        self.num_views = len(ds_list)
        self.learners = []
        self.filtering_history = None

    def fit(self, max_iter=10):

        for i in xrange(self.num_periods):

            if i > 0:
                for ds in self.ds_list:
                    ds.refresh()

            periodic_dss = [PM(ds, self.pl, self.num_periods, i+1)
                            for ds in self.ds_list] 
            gs_list = [self.get_gs() for i in xrange(self.num_views)]
            learner = self.get_learner(
                self.k, periodic_dss, gs_list, self.verbose)

            learner.fit(max_iter=max_iter)
            self.learners.append(learner)

            # Merge the periodic filtering histories correctly
            self._set_merged_filtered_Xs()

    def _set_merged_filtered_Xs(self):

        filtered_X_list = [l.get_status()['filtering_history']
                           for l in self.learners]
        finished = [False] * self.num_views
        period_count = 0
        
        while not all(finished):
            if self.filtering_history is None:
                self.filtering_history = [filtered_X_list[0][i][:self.pl,:]
                                          for i in xrange(self.num_views)]
                
                finished = self._update_merged_filtered_Xs(
                    filtered_X_list[1:], period_count)
            else:
                finished = self._update_merged_filtered_Xs(
                    filtered_X_list, period_count)

            period_count += 1

    def _update_merged_filtered_Xs(self, 
        filtered_X_list, period_count):

        finished = [False] * self.num_views

        for fX in filtered_X_list:
            for i in xrange(self.num_views):
                begin = self.pl * period_count
                end = begin + self.pl

                if end > fX[i].shape[0]:
                    current = self.filtering_history[i]
                    new = fX[i][begin:end]

                    self.filtering_history[i] = np.vstack([current, new])
                else:
                    finished[i] = True

        return finished

    def get_status(self):

        return {
            'k': self.k,
            'ds_list': self.ds_list,
            'get_learner': self.get_learner,
            'period_length': self.pl,
            'num_periods': self.num_periods,
            'verbose': self.verbose,
            'get_gs': self.get_gs,
            'num_views': self.num_views,
            'learners': self.learners}
