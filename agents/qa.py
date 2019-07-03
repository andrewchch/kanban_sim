import logging
from agents.agent import Agent


class QA(Agent):

    QA_FIFO_STRATEGY = 0

    strategies = [
        'FIFO'
    ]

    def __init__(self, name, env, strategy, work_pile, current_release):
        self.name = name
        self.env = env
        self.strategy = strategy
        self.work_pile = work_pile
        self.current_release = current_release

    def run(self):
        """
        A QA takes a case off the QA pile, works on it for a period of time proportional to its size then marks it as
        finished. They can only work on one at a time.
        """
        while True:
            # If a release is in test, we're regression testing it so don't work on cases
            if not self.current_release is None and self.current_release.in_test:
                logging.debug('%s Doing regression testing at %s' % (self.name, self.env.now))
                yield self.env.timeout(1)
                continue

            # Get a work item off the QA backlog
            logging.debug('%s requesting work at %s' % (self.name, self.env.now))
            work = yield self.work_pile.get()
            logging.debug('%s got work %s at %s' % (self.name, work, self.env.now))

            size = work.size()
            yield self.env.timeout(size)
            work.finish(by=self.name)
            logging.info('%s finished %s at %s' % (self.name, work, self.env.now))

