class PeriodicMetaLearner:

    def __init__(self, 
        learner_factory, 
        period_length, 
        num_periods):

        self.learner_factory = learner_factory
        self.period_length = period_length
        self.num_periods = num_periods

        self.learners = []

    def fit(self):

        print 'Stuff'

    def get_status(self):

        return {}
