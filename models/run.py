class Run:
    """
    A set of output data from a given run using possibly different parameters and strategies
    """
    def __init__(self, params=None, data=None):
        self.params = params
        self.data = data

    def run_data(self, name=None):
        return self.data[name]


