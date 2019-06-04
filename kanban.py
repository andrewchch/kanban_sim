"""
Ideas:

1. Add extra queues for different stages (code review)
1a. Developers can choose whether to code review or do dev work (how much time is actually spent doing code reviews in practice?)
1b. Developers can also choose to do QA work (but they will be half as quick)
2. Model different behaviours around choosing what to work on (ignore priorities) (DONE)
3. Implement case priorities or business value (DONE)
4. Add a release phase, either when a fixed number of cases are done or when a certain amount of time has passed. The idea is to see
how released value varies accordingly. (DONE)
5. Track flow efficiency of each case
6. For each actor, track whether they are starved or blocked in a given turn
7. Track time spent in different states
8. Maybe model choices around severity and priority? Would be ideal to do some analysis of the likelihood of real cases
getting fixed with X days based on severity and priority
"""
import simpy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy

from tqdm import tqdm
from datetime import datetime

RANDOM_SEED = 42
SIM_TIME = 1000  # Simulation time in days
DEVELOPER_CAPACITY = 4
QA_CAPACITY = 1
NUM_DEVELOPERS = 8
NUM_QA = 2
NUM_BA = 1
NUM_RUNS = 200
MAX_QA_PILE = 10
MAX_DEV_PILE = 20
MAX_REVIEW_PILE = 20
QA_WORK_PER_CASE = 4
REGRESSION_TEST_WORK_PER_RELEASE = 10
ANALYSIS_WORK_PER_CASE = 2
NUM_SOURCE_CASES = 300
NUM_INITIAL_CASES = 100
MAX_RELEASE_CASES = 5
NEW_CASE_INTERVAL = 5
ANALYSIS_PILE_MIN = 5

# Dev strategy options
HIGHEST_COST_FIRST = 0  # red
HIGHEST_VALUE_FIRST = 1  # pink
BIGGEST_FIRST = 2  # green
SMALLEST_FIRST = 3  # blue
HIGHEST_VALUE_TO_COST_RATIO_FIRST = 4  # orange
RANDOM = 5  # black

# BA strategies
BA_FIFO_STRATEGY = 0
BA_HIGHEST_VALUE_TO_COST_RATIO_FIRST = 1

logging.basicConfig(level=logging.ERROR,
                    format='%(message)s',)


class Work:
    """ A body of work to be done for a case stage """
    def __init__(self, name='Work', size=0, env=None):
        self.work_to_do = size
        self.work_done = 0
        self.done_at = None
        self.name = name
        self.env = env

    def set_env(self, env):
        self.env = env

    def is_done(self):
        return self.work_to_do == self.work_done

    def do_work(self):
        if self.work_done < self.work_to_do:
            self.work_done += 1

    def finish(self):
        self.work_done = self.work_to_do
        self.done_at = self.env.now

    def size(self):
        return self.work_to_do

    def __str__(self):
        return '%s work of size %d with %d done' % (self.name, self.work_to_do, self.work_done)


class Run:
    """
    A set of output data from a given run using possibly different parameters and strategies
    """
    def __init__(self, params=None, data=None):
        self.params = params
        self.data = data

    def run_data(self, name=None):
        return self.data[name]

class Case:
    """ A task to be done consisting of a series of pieces of Work """
    issuetype = 'Bug'
    index = 0

    def __init__(self, index=index, size=0, value=0, env=None):
        self.states = [
            Work(name='Analysis', size=ANALYSIS_WORK_PER_CASE),
            Work(name='Dev', size=size),
            Work(name='QA', size=QA_WORK_PER_CASE)
        ]
        self.index = index
        self.size = size
        self.value = value
        self.env = env
        self.state = 0
        self.start_time = self.env is not None and self.env.now or 0
        self.end_time = 0
        self.release_time = 0
        self.release = None

    def set_env(self, env):
        self.env = env
        self.start_time = self.env.now
        for work in self.states:
            work.set_env(env)

    def is_done(self):
        """ The final state is finished """
        return self.state == len(self.states) - 1 and self.current_state().is_done()

    def current_state(self):
        return self.states[self.state]

    def cycle_time(self):
        return max(self.end_time - self.start_time, 0)

    def finish_current(self):
        """
        Finish all work on the current state. Move to the next state but
        return the state we were just working on, because whoever is working on it might need to do something
        with it.
        """
        if self.start_time == 0:
            self.start_time = self.env.now
        _state = self.current_state()
        _state.finish()
        if self.is_done():
            self.end_time = self.env.now
        return _state

    def do_work(self):
        """
        Do work on the current state. If the work is all done on the current state, move to the next state but
        return the state we were just working on, because whoever is working on it might need to do something
        with it.
        """
        if self.start_time == 0:
            self.start_time = self.env.now
        _state = self.current_state()
        _state.do_work()
        if self.is_done():
            self.end_time = self.env.now
        return _state

    def goto_next_state(self):
        """
        Go to the next state and return the new state if there's still work to do or None if the case is finished.
        :return: current state
        """
        if not self.is_done():
            self.state += 1
            return self.current_state()
        else:
            return None

    def do_release(self):
        self.release_time = self.env.now

    def get_current_value(self):
        """
        Value can be positive or negative and accumulates over time, from the start of the cycle to the point it is released.
        When a case is added to the backlog, the cycle starts, so the flow probably needs some time to get into a steady state.
        Initial approach is a simple linear relationship, current value is a product of initial value and time
        :return:
        """
        # If the cost is negative, it accumulates until it's fixed and then returns to 0. If positive,
        # it continues to accumulate after it finishes
        if self.value < 0:
            if 0 < self.release_time < self.env.now:
                return 0
            else:
                return self.value * (self.env.now - self.start_time)
        else:
            if 0 < self.release_time < self.env.now:
                return self.value * (self.env.now - self.release_time)
            else:
                return 0


    def __str__(self):
        return 'Case %d of size %d' % (self.index, self.size)


class Release(object):
    def __init__(self, env=None):
        self._in_test = False
        self.released = False
        self.released_time = None
        self.cases = []
        self.env = env

    @property
    def in_test(self):
        return self._in_test

    @in_test.setter
    def in_test(self, val):
        self._in_test = val

    def release(self):
        self._in_test = False
        self.released = True
        self.released_time = self.env.now
        for _case in self.cases:
            _case.do_release()

    def add_case(self, case):
        self.cases.append(case)
        case.release = self

    def __str__(self):
        return 'Release with %d cases, in_test = %s, released_time = %s' % (len(self.cases), self._in_test, self.released_time)


def monitor(env):
    while True:
        #if len(analysis_pile.items) == 0 and len(dev_pile.items) == 0 and len(qa_pile.items) == 0 and len(done_pile.items) == 0:
        #    env.exit(1)

        total_value = 0
        for case in cases:
            total_value += case.get_current_value()

        total_values.append(total_value)

        analysis_size.append(len(analysis_pile.items))
        dev_size.append(len(dev_pile.items))
        dev_put_queue_size.append(len(dev_pile.put_queue))
        qa_size.append(len(qa_pile.items))
        done_size.append(len(done_pile.items))
        release_size.append(len(released_pile.items))

        yield env.timeout(1)


def ba(name, env, strategy):
    """A BA takes a case off the Analysis pile, works on it for a period of time proportional to its size, then
    moves it to the Dev pile. They can only work on one at a time.
    """
    while True:
        # Apply a strategy to grooming the analysis pile (backlog)
        if strategy == BA_HIGHEST_VALUE_TO_COST_RATIO_FIRST:
            analysis_pile.items.sort(key=lambda x: x.value/x.size, reverse=True)

        # Get a case off the Analysis backlog
        logging.debug('%s requesting a case at %s' % (name, env.now))
        case = yield analysis_pile.get()

        # Work on it until it's done, then move it to the Dev pile
        size = case.current_state().size()

        yield env.timeout(size)
        case.finish_current()
        case.goto_next_state()
        dev_pile.put(case)
        logging.info('%s finished %s at %s, pushing to Dev' % (name, case, env.now))


def developer(name, env, strategy=HIGHEST_COST_FIRST):
    """A developer takes a case off the backlog, works on it for a period of time proportional to its size, then
    moves it to the done pile.

    Developers can have multiple cases on the go (up to LIMIT), and each day they can choose to work on one case or
    another until it is done.
    """
    my_pile = simpy.Store(env, DEVELOPER_CAPACITY)
    started = False

    while True:
        #logging.debug('%s %s' % (len(my_pile.items), len(dev_pile.items)))
        # Only try to fetch a new case if this developer has spare capacity
        if len(my_pile.items) < DEVELOPER_CAPACITY and len(dev_pile.items) > 0:

            # Get a case off the backlog
            logging.debug('%s requesting a case at %s' % (name, env.now))
            case = yield dev_pile.get()

            # If there was a case to get, add it to my pile
            if case:
                started = True
                logging.debug('%s got %s at %s' % (name, case, env.now))
                my_pile.put(case)

        # Either no more items to work on since we started or we haven't started yet
        if len(my_pile.items) == 0:
            if started:
                return 0
            else:
                yield env.timeout(1)
                continue

        # Choose which case to work on
        if strategy == RANDOM:
            # Strategy #1: Choose randomly
            np.random.seed(datetime.now().microsecond)
            case_to_work_on = my_pile.items[np.random.randint(len(my_pile.items))]
        elif strategy in (BIGGEST_FIRST, SMALLEST_FIRST):
            if strategy == SMALLEST_FIRST:
                case_index = 0
            else:
                case_index = -1
            case_to_work_on = sorted(my_pile.items, key=lambda x: x.size)[case_index]
        elif strategy in (HIGHEST_COST_FIRST, HIGHEST_VALUE_FIRST):
            # Strategy #2: Work on the highest cost case first (max negative value)
            # Strategy #3: Work on the highest value case first (max positive value)
            if strategy == HIGHEST_COST_FIRST:
                case_index = 0
            else:
                case_index = -1
            case_to_work_on = sorted(my_pile.items, key=lambda x: x.value)[case_index]
        elif strategy == HIGHEST_VALUE_TO_COST_RATIO_FIRST:
            case_to_work_on = sorted(my_pile.items, key=lambda x: x.value/x.size)[-1]
        else:
            raise Exception('Invalid strategy: %s' % strategy)

        # Work on the case
        if not case_to_work_on.is_done():
            current_state = case_to_work_on.current_state()

            if current_state.is_done():
                # If the dev work was just finished, move the case to the QA pile
                logging.info('%s finished %s work on %s at %s' % (name, current_state.name, case_to_work_on, env.now))
                if current_state.name == 'Dev':
                    # Move the case to the QA pile and step to the next state
                    case_to_work_on.goto_next_state()
                    my_pile.items.remove(case_to_work_on)
                    qa_pile.put(case_to_work_on)
                    logging.info('%s moved %s to QA at %s' % (name, case_to_work_on, env.now))

            case_to_work_on.do_work()
            logging.debug('%s worked on %s at %s' % (name, case_to_work_on, env.now))
        else:
            logging.debug('%s finished %s at %s' % (name, case_to_work_on, env.now))

        # Do this every day
        yield env.timeout(1)


def qa(name, env, current_release=None):
    """
    A QA takes a case off the QA pile, works on it for a period of time proportional to its size, then
    moves it to the done pile. They can only work on one at a time.
    """
    while True:
        # If a release is in test, we're regression testing it so don't work on cases
        if not current_release is None and current_release.in_test:
            logging.debug('%s Doing regression testing at %s' % (name, env.now))
            yield env.timeout(1)
            continue

        # Get a case off the QA backlog
        logging.debug('%s requesting a case at %s' % (name, env.now))
        case = yield qa_pile.get()
        logging.debug('%s got case %s at %s' % (name, case, env.now))

        size = case.current_state().size()
        yield env.timeout(size)
        case.finish_current()
        case.goto_next_state()
        done_pile.put(case)
        logging.info('%s finished %s at %s' % (name, case, env.now))


def releaser(name, env, current_release=None, releases=None):
    """
    A Releaser takes cases off the done pile and assigns them to a release. QA people can then do regression testing release
    after which all the cases in the release can be marked as released.
    """
    while True:
        # Get a case off the Done pile and add it to a release
        logging.debug('%s requesting a case at %s' % (name, env.now))
        case = yield done_pile.get()
        logging.debug('%s got case %s at %s' % (name, case, env.now))

        # If the release has enough cases on it (or enough time has passed, todo),
        # mark it as released, mark all associated cases as released and update value
        if current_release is None:
            current_release = Release(env)

        if len(current_release.cases) > MAX_RELEASE_CASES:

            # Flag the shared release as in test
            current_release.in_test = True

            # Wait for the regression test interval
            yield env.timeout(REGRESSION_TEST_WORK_PER_RELEASE)
            current_release.in_test = False

            # Add the finished release to the list of completed releases and start a new release
            current_release.release()
            releases.append(current_release)
            for case in current_release.cases:
                released_pile.put(case)

            current_release = Release(env)
            logging.info('%s released %s at %s' % (name, current_release, env.now))
        else:
            # Just add the case to the current release
            current_release.add_case(case)


def sourcer(name, env):
    """
    The job of the sourcer is just to generate new cases by adding them from the backlog periodically.
    """
    while True:
        # Get a case off the Source backlog
        if len(analysis_pile.items) <= ANALYSIS_PILE_MIN:
            logging.debug('%s fetching a source case at %s' % (name, env.now))
            case = yield source_pile.get()

            # Work on it until it's done, then move it to the Dev pile
            yield env.timeout(NEW_CASE_INTERVAL)
            analysis_pile.put(case)
            logging.info('%s pushing %s to analysis at %s' % (name, case, env.now))
        else:
            yield env.timeout(1)


def plot_ecdf(data, color='black'):
    _ = plt.plot(np.sort(data), np.arange(1, len(data) + 1) / len(data), marker='.', linestyle='none', alpha=0.01, color=color)
    return _


# Create a repeatable backlog of cases that have binomially distributed sizes and normally distributed values
logging.info('Creating standard cases')
np.random.seed(RANDOM_SEED)

standard_cases = []
sizes = (np.random.binomial(10, 0.2, size=NUM_SOURCE_CASES) + 1) * 10
values = (np.random.normal(0, 1, size=NUM_SOURCE_CASES)) * 10
for idx, size in enumerate(sizes):
    standard_cases.append(Case(index=idx, size=max(size, 1), value=values[idx]))  # value = size

# Do multiple executions of this model where behaviour of agents varies
runs = []
"""
analysis_size_runs = []
dev_size_runs = []
review_size_runs = []
dev_put_queue_runs = []
qa_size_runs = []
done_size_runs = []
release_size_runs = []
cycle_time_runs = []
total_value_runs = []
"""
releases = []
cases = []
current_release = None  # a global resource, can only be one release active at a time
sourcers = []

for run in tqdm(range(NUM_RUNS)):
    logging.debug('---- Run %s -----' % run)
    # Create environment and start processes
    analysis_size = []
    dev_size = []
    review_size = []
    dev_put_queue_size = []
    qa_size = []
    done_size = []
    release_size = []
    cycle_times = []
    total_values = []

    # Choose a random strategy for each run
    strategy = np.random.randint(6)
    ba_strategy = np.random.randint(2)

    env = simpy.Environment()

    # Create piles
    source_pile = simpy.Store(env, len(sizes))
    analysis_pile = simpy.Store(env, len(sizes))
    dev_pile = simpy.Store(env, MAX_DEV_PILE)
    review_pile = simpy.Store(env, MAX_REVIEW_PILE)
    qa_pile = simpy.Store(env, MAX_QA_PILE)
    done_pile = simpy.Store(env)
    released_pile = simpy.Store(env)

    # Create a Run to store params
    run = Run(params={
            'strategy': strategy,
            'ba_strategy': ba_strategy
        })

    # Create a pile of all source cases
    for case in standard_cases[NUM_INITIAL_CASES:len(standard_cases)]:
        _case = copy.deepcopy(case)
        _case.set_env(env)
        source_pile.put(_case)

    # Create a pile of cases to get the system to a steady state. These will initially go into the analysis pile.
    for case in standard_cases[:NUM_INITIAL_CASES]:
        _case = copy.deepcopy(case)
        _case.set_env(env)
        analysis_pile.put(_case)

    cases = copy.copy(analysis_pile.items)

    # Create some developers (who will do both development and reviews) with a random strategy
    developers = []
    for i in range(NUM_DEVELOPERS):
        developers.append(env.process(developer('Developer %d' % i, env, strategy=strategy)))

    # Create some QAs
    qas = []
    for i in range(NUM_QA):
        qas.append(env.process(qa('QA %d' % i, env, current_release)))

    # Create some BAs
    bas = []
    for i in range(NUM_BA):
        bas.append(env.process(ba('BA %d' % i, env, ba_strategy)))

    # Create a Releaser
    releasers = [env.process(releaser('Releaser 0', env, current_release, releases))]

    # Create a case Sourcer
    sourcers = [env.process(sourcer('Sourcer 0', env))]

    # add monitoring
    env.process(monitor(env))

    # Execute!
    env.run(until=SIM_TIME)

    # Store cycle time values
    for case in cases:
        cycle_times.append(case.cycle_time())

    # Store stats for this run
    run.data = {
        'analysis': analysis_size,
        'dev': dev_size,
        'review': review_size,
        'qa': qa_size,
        'done': done_size,
        'release': release_size,
        'cycle_time': cycle_times,
        'total_value': total_values
    }

    runs.append(run)


# Counts in each state
for i, data in enumerate([x.run_data('analysis') for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='blue')
for i, data in enumerate([x.run_data('dev') for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='red')
for i, data in enumerate([x.run_data('qa') for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='green')
for i, data in enumerate([x.run_data('done') for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='orange')
for i, data in enumerate([x.run_data('release') for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='black')
plt.show()

# Cycle time ecdf
colors = ['red', 'pink', 'green', 'blue', 'orange', 'black']
line_styles = ['-', '--', '-.', ':']  # solid, dash, dash-dot, dot

for run in runs:
    color = colors[run.params['strategy']]
    _ = plot_ecdf(run.run_data('cycle_time'), color=color)
plt.show()

# Total value plot
for run in runs:
    color = colors[run.params['strategy']]
    line_style = line_styles[run.params['ba_strategy']]
    _ = plt.plot(run.run_data('total_value'), alpha=0.1, color=color, linestyle=line_style)
plt.show()