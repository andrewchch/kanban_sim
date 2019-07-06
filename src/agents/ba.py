import logging
from agents.agent import Agent


class BA(Agent):

    BA_FIFO_STRATEGY = 0
    BA_HIGHEST_VALUE_TO_COST_RATIO_FIRST = 1

    strategies = [
        'FIFO',
        'Highest value-to-cost ratio first'
    ]

    def __init__(self, name, env, strategy, work_pile):
        self.name = name
        self.env = env
        self.strategy = strategy
        self.work_pile = work_pile

    def run(self):
        while True:
            # Apply a strategy to grooming the analysis pile (backlog)
            if self.strategy == self.BA_HIGHEST_VALUE_TO_COST_RATIO_FIRST:
                self.work_pile.items.sort(key=lambda x: x.case.value / x.case.size, reverse=True)

            # Get a work item off the Analysis backlog
            logging.debug('%s requesting a case at %s' % (self.name, self.env.now))
            work = yield self.work_pile.get()

            # Work on it until it's done
            size = work.size()

            yield self.env.timeout(size)
            work.finish(by=self.name)
            logging.info('%s finished %s at %s' % (self.name, work, self.env.now))
