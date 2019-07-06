import numpy.random as npr


class Agent:
    _strategies = []

    @classmethod
    def random_strategy(cls):
        return npr.choice(len(cls.strategies))

    @classmethod
    def get_strategies(cls):
        return cls.strategies
