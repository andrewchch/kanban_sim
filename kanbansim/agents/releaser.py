import logging
from agents.agent import Agent
from models import Release


class Releaser(Agent):

    current_release: Release
    MAX_RELEASE_CASES = 5
    REGRESSION_TEST_WORK_PER_RELEASE = 10

    strategies = [
        'Release when max cases reached',
    ]

    def __init__(self, name, env, current_release=None, done_pile=None, releases=None, released_pile=None, standard_cases=None):
        self.name = name
        self.env = env
        self.releases = releases
        self.released_pile = released_pile
        self.done_pile = done_pile
        self.current_release = current_release
        self.standard_cases = standard_cases

    def run(self):
        """
        A Releaser takes cases off the done pile and assigns them to a release. QA people can then do regression
        testing release after which all the cases in the release can be marked as released.
        """
        while True:
            # Get a case off the Done pile and add it to a release
            logging.debug('%s requesting a case at %s' % (self.name, self.env.now))
            case = yield self.done_pile.get()
            logging.debug('%s got %s at %s' % (self.name, case, self.env.now))

            # If the release has enough cases on it or this is the balance of the releasable cases
            # mark it as released, mark all associated cases as released and update value
            # todo: or add a time limit
            if self.current_release is None:
                self.current_release = Release(self.env)

            logging.debug('case: %s, released_pile: %s, current_release: %s at %s' % (case.name,
                                                                                      len(self.released_pile.items),
                                                                                      len(self.current_release.cases),
                                                                                      self.env.now))

            # Just add the case to the current release
            self.current_release.add_case(case)

            # Do a release if necessary
            if len(self.current_release.cases) >= self.MAX_RELEASE_CASES or\
                    (len(self.current_release.cases) +
                     len(self.released_pile.items)) == len(self.standard_cases):

                # Flag the shared release as in test
                self.current_release.in_test = True

                # Wait for the regression test interval
                yield self.env.timeout(self.REGRESSION_TEST_WORK_PER_RELEASE)
                self.current_release.in_test = False

                # Add the finished release to the list of completed releases and start a new release
                self.current_release.release()
                self.releases.append(self.current_release)
                for case in self.current_release.cases:
                    self.released_pile.put(case)

                self.current_release = Release(self.env)
                logging.info('%s released %s at %s' % (self.name, self.current_release, self.env.now))
