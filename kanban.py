"""
DONE:
2. Model different behaviours around choosing what to work on (ignore priorities) (DONE)
3. Implement case priorities or business value (DONE)
4. Add a release phase, either when a fixed number of cases are done or when a certain amount of time has passed. The idea is to see
how released value varies accordingly. (DONE)
5. Allow the varying of any parameter in the model so we can do "hyperparameter tuning" (or equivalent), e.g., vary the number of QAs
to see what the impact on value delivered is
6. Remove the clunky connection between current state and the pile that a case is on to control who can pick it up next. Feels
clumsy and that we should only need to use one state variable to control this.
7. Factor in intangibles - expensive tasks that generate value at an increasing rate over time, so are effectively long-term
investments
"""
import sys
import simpy
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy

from tqdm import tqdm
from datetime import datetime
from functools import reduce
from pubsub import pub

RANDOM_SEED = 42
SIM_TIME = 1000  # Simulation time in days
DEVELOPER_CAPACITY = 4
DEVELOPER_REVIEW_CAPACITY = 2
QA_CAPACITY = 1
NUM_DEVELOPERS = 8
NUM_QA = 2
NUM_BA = 1
NUM_RUNS = 200
#NUM_RUNS = 1
MAX_QA_PILE = 10
MAX_DEV_PILE = 20
MAX_REVIEW_PILE = 20
QA_WORK_PER_CASE = 4
MERGE_WORK_PER_CASE = 1
REVIEW_WORK_RATIO = 0.5

REGRESSION_TEST_WORK_PER_RELEASE = 10
ANALYSIS_WORK_PER_CASE = 2
NUM_SOURCE_CASES = 300
NUM_INITIAL_CASES = 100
#NUM_SOURCE_CASES = 20
#NUM_INITIAL_CASES = 10

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

# Review pile strategy options (how the dev chooses from their pile of items to review)
REVIEW_HIGHEST_V2CR_FIRST = 0
REVIEW_OLDEST_FIRST = 1
REVIEW_YOUNGEST_FIRST = 2
REVIEW_HIGHEST_COST_FIRST = 3
REVIEW_HIGHEST_VALUE_FIRST = 4

# Dev choice strategy (how the dev decides whether to work on dev work or review work)
# todo: add other more intelligent strategies, e.g., if a review is older than X, do that first
DEV_REVIEWS_FIRST = 0
DEV_RANDOM_CHOICE = 1
DEV_DEV_WORK_FIRST = 2

# BA strategies
BA_FIFO_STRATEGY = 0
BA_HIGHEST_VALUE_TO_COST_RATIO_FIRST = 1

# Work names
WORK_ANALYSIS = 'Analysis'
WORK_DEV = 'Dev'
WORK_REVIEW = 'Review'
WORK_QA = 'QA'
WORK_MERGE = 'Merge'

logging.basicConfig(level=logging.ERROR,
                    format='%(message)s',)


class Workflow:
    """
    A workflow containing some work items (or potentially other workflows) to do and how to manage them, e.g., the
    assignee can do them or must delegate them to someone else. Workflow steps can contain other Workflow steps so an
    entire Workflow can be defined by one WorkflowStep
    """
    def __init__(self, work_items=[], delegate=False, sequential=True, case=None, workflow=None):
        assert (work_items is not None and type(work_items) == list and len(work_items) > 0), "Need to specify some work items to do"

        self.work_items = work_items

        # Let each item know that we are its parent
        for item in self.work_items:
            item.set_workflow(self)

        self.delegate = delegate
        self.sequential = sequential  # Work items in this step can be done in parallel or must be done sequentially
        self.assignee = None
        self.env = None
        self.index = -1
        self.name = None  # todo: future: workflows can be named for flow control
        self.case = case
        self.workflow = workflow
        self.start_time = None
        self.started = False

    def current_step(self):
        return self.work_items[self.index]

    def set_workflow(self, workflow):
        """
        Set the parent workflow
        :param workflow:
        :return:
        """
        self.workflow = workflow

    def set_case(self, case):
        """
        Set the parent case
        :param case:
        :return:
        """
        self.case = case
        # Let each item know its case
        for item in self.work_items:
            item.set_case(case)

    def set_start_time(self, time):
        """
        Set the start time of the workflow and parent if it hasn't already been set
        :param time:
        :return:
        """
        if not self.started:
            if self.workflow is not None:
                self.workflow.set_start_time(time)
            if self.case is not None:
                self.case.set_start_time(time)

            self.start_time = time
            self.started = True

    def get_next_step(self):
        """
        Go to the next state and return the corresponding workflow item if there's still work to do or None if the case
        is finished. If we're already on the last step of the workflow, return None.
        :return: current step as a list of one or more work items that can be run in parallel
        """
        if self.sequential:
            if self.index < len(self.work_items):
                self.index += 1
                item = self.work_items[self.index]

                # If the next item is a Workflow, return its next Work item
                if issubclass(type(item), Workflow):
                    return item.get_next_step()
                elif type(item) is Work:
                    return [item]
        elif self.index < 0:
            # Can run all work items in parallel so return all of them
            self.index = 0
            next_items = []
            for item in self.work_items:
                if issubclass(type(item), Workflow):
                    next_items.extend(item.get_next_step())
                elif type(item) is Work:
                    next_items.append(item)

            return next_items

        return None

    def set_env(self, env):
        self.env = env
        for item in self.work_items:
            item.set_env(env)

    def add_item(self, work):
        assert work is not None, "Invalid work item specified"
        self.work_items.append(work)

    def is_done(self):
        """
        Return whether all work items have been done
        """
        return reduce((lambda x, y: x and y), [w.is_done() for w in self.work_items])

    def work_is_finished(self):
        """
        A signal from an associated work item that it has been finished.
        :return:
        """
        if self.is_done():
            # If we have a parent workflow then notify it, otherwise if the workflow is now finished as well,
            # now's the time to notify anyone doing dispatch.
            if self.workflow is not None:
                self.workflow.work_is_finished()
            else:
                pub.sendMessage('case_is_done', case=self.case)
        else:
            # Otherwise, get the next step and put it on the dispatch pile
            items = self.get_next_step()
            if items is not None:
                for item in items:
                    dispatch_pile.append(item)

    def finish_current(self, by):
        """
        Finish all work on the current state. Move to the next state but
        return the state we were just working on, because whoever is working on it might need to do something
        with it.
        """
        if self.start_time == 0:
            self.start_time = self.env.now
        _state = self.current_step()
        _state.finish(by=by)
        if self.is_done():
            self.end_time = self.env.now
        return _state

    def assign_to(self, to):
        self.assignee = to

    def size(self):
        """
        Returns the sum of sizes of all work items
        :return:
        """
        return sum([w.size() for w in self.work_items])


def developer_of_work_item(item):
    """
    Get the developer of the dev work associated with this review or merge item
    :param item:
    :return:
    """
    assert type(item) is Work, "Not a work item"

    # Get the grandparent workflow
    dev_workflow = None

    if item.name == WORK_REVIEW:
        review_workflow = item.workflow
        if review_workflow is not None:
            dev_workflow = review_workflow.workflow
    elif item.name == WORK_MERGE:
        dev_workflow = item.workflow

    if dev_workflow is not None and type(dev_workflow) is DevWorkflow:
        dev_work = dev_workflow.work_items[0]
        return dev_work.work_done_by

    return None


class DevWorkflow(Workflow):
    """
    Custom workflow to connect the dev work and review steps. Assumes that the workflow consists of

    - Dev work item
    - Parallel workflow with two review items
    """
    def __init__(self, *args, **kwargs):
        super(DevWorkflow, self).__init__(*args, **kwargs)
        assert len(self.work_items) == 4, "Incorrect number of work items for DevWorkflow, should be 2 "
        assert type(self.work_items[0]) is Work, "Non-DEV first work item for DevWorkflow"
        assert type(self.work_items[1]) is Workflow, "Non-Workflow second work item for DevWorkflow"
        assert type(self.work_items[2]) is Work, "Non-Work third work item for DevWorkflow"
        assert type(self.work_items[3]) is Work, "Non-Work fourth work item for DevWorkflow"


class Work:
    """
    A body of work to be done for a workflow step. Assumption is that all work is done by one actor.
    """
    def __init__(self, name='Work', size=0, env=None, workflow=None, case=None):
        assert size is not None, "Size must be greater than zero"
        self.work_to_do = size
        self.work_done = 0
        self.name = name
        self.env = env
        self.work_done_by = None
        self.assignee = None
        self.workflow = workflow
        self.case = case
        self.started = False
        self.start_time = None
        self.end_time = None

    def set_workflow(self, workflow):
        """
        Set the parent workflow
        :param workflow:
        :return:
        """
        self.workflow = workflow

    def set_case(self, case):
        """
        Set the parent case
        :param case:
        :return:
        """
        self.case = case

    def set_env(self, env):
        self.env = env

    def is_done(self):
        return self.work_to_do == self.work_done

    def done_by(self):
        return self.work_done_by

    def assign_to(self, _assignee):
        self.assignee = _assignee

    def assigned_to(self):
        return self.assignee

    def do_work(self, by=None):
        """
        Do some work (must have an assignee first)
        :return:
        """
        assert (by or self.assignee) is not None, "Noone specified to work on the work item"

        # Set the start time if it hasn't already been set
        if not self.started:
            self.start_time = self.env.now
            if self.workflow is not None:
                self.workflow.set_start_time(self.env.now)
            self.started = True

        self.work_done_by = by or self.assignee
        if self.work_done < self.work_to_do:
            self.work_done += 1

        if self.is_done():
            self.end_time = self.env.now
            # Signal to the workflow that we're finished
            logging.info('%s finished %s at %s' % (self.work_done_by, self, self.env.now))
            self.workflow.work_is_finished()

    def finish(self, by=None):
        """
        Flag the work as having been done by the assignee and then unassign the work.
        :param by:
        :return:
        """
        assert (by or self.assignee) is not None, "Noone specified to finish the workstep"

        # Set the start time if it hasn't already been set
        if not self.started:
            self.start_time = self.env.now
            if self.workflow is not None:
                self.workflow.set_start_time(self.env.now)
            self.started = True

        self.work_done_by = by or self.assignee
        self.assignee = None
        self.work_done = self.work_to_do
        self.end_time = self.env.now

        # Signal to the workflow that we're finished
        self.workflow.work_is_finished()

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
    """
    A task to be done dictated by a Workflow. A case will have a primary assignee by virtue of being
    on that actor's backlog, but individual work items (e.g., review work) can be picked up by other actors.
    A case is primarily used to track overall metrics, e.g., cycle time, current value
    """
    issuetype = 'Bug'  # todo: implement case types
    index = 0

    def __init__(self, size=0, value=0, env=None, name=None):
        assert size is not None, "Size must be greater than zero"
        self.workflow = Workflow(
            work_items=[
                Work(name=WORK_ANALYSIS, size=ANALYSIS_WORK_PER_CASE),
                DevWorkflow(work_items=[
                    Work(name=WORK_DEV, size=size),
                    Workflow(
                        work_items=[
                            Work(name=WORK_REVIEW, size=int(size * REVIEW_WORK_RATIO)),
                            Work(name=WORK_REVIEW, size=int(size * REVIEW_WORK_RATIO))
                        ],
                        sequential=False
                    ),
                    Work(name=WORK_QA, size=QA_WORK_PER_CASE),
                    Work(name=WORK_MERGE, size=MERGE_WORK_PER_CASE)
                ]),
            ],
            sequential=True
        )

        self.workflow.set_case(self)
        self.size = size
        self.value = value
        self.env = env
        self.end_time = 0
        self.release_time = 0
        self.release = None
        self.name = name
        self.start_time = None
        self.started = None

    def set_env(self, env):
        self.env = env
        self.start_time = self.env.now
        for item in self.workflow.work_items:
            item.set_env(env)

    def set_start_time(self, time):
        """
        Set the start time of the case if it hasn't already been set
        :param time:
        :return:
        """
        if not self.started:
            self.start_time = time
            self.started = True

    def is_done(self):
        """ Same as the workflow being finished """
        return self.workflow.is_done()

    def cycle_time(self):
        return max(self.release_time - self.start_time, 0)

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
        return 'Case %s of size %d' % (self.name, self.size)


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


def monitor(env, finisher):
    while True:
        if len(released_pile.items) == len(standard_cases):
            # Trigger the finisher event to succeed which will end the sim
            finisher.succeed()

        # Calculate the total value of all cases that are in the flow.
        total_value = 0
        for case in progress_pile.items:
            total_value += case.get_current_value()

        total_values.append(total_value)

        # Track numbers of work items in different states (which is not the same as cases in different workflow
        # states but we'll add that next)
        # todo: track cases in each workflow state. Tricky thing is a case can potentially be in many "states"
        # todo: if there are multiple work items in action at once. Might be easier to count unique combinations of
        # todo: case and workflow type.
        analysis_size.append(len(work_piles[WORK_ANALYSIS].items))
        dev_size.append(len(work_piles[WORK_DEV].items))
        dev_put_queue_size.append(len(work_piles[WORK_DEV].put_queue))
        qa_size.append(len(work_piles[WORK_QA].items))
        merge_size.append(len(work_piles[WORK_MERGE].items))
        done_size.append(len(done_pile.items))
        release_size.append(len(released_pile.items))

        yield env.timeout(1)


def ba(name, env, strategy):
    """A BA takes a case off the Analysis pile, works on it for a period of time proportional to its size
    the marks it as done. They can only work on one work item at a time.

    The BA has knowledge of the value of a case in order to be able to prioritise cases according to value and the
    current strategy.
    """
    while True:
        # Apply a strategy to grooming the analysis pile (backlog)
        if strategy == BA_HIGHEST_VALUE_TO_COST_RATIO_FIRST:
            work_piles[WORK_ANALYSIS].items.sort(key=lambda x: x.case.value/x.case.size, reverse=True)

        # Get a work item off the Analysis backlog
        logging.debug('%s requesting a case at %s' % (name, env.now))
        work = yield work_piles[WORK_ANALYSIS].get()

        # Work on it until it's done
        size = work.size()

        yield env.timeout(size)
        work.finish(by=name)
        logging.info('%s finished %s at %s' % (name, work, env.now))


def developer(name, env, strategy=HIGHEST_COST_FIRST):
    """A developer takes a work item off the backlog, works on it for a period of time proportional to its size, then
    marks it as done.

    Developers can have multiple cases on the go (up to LIMIT), and each day they can choose to work on one case or
    another until it is done.

    The Developer has knowledge of the value of a case in order to be able to prioritise cases according to value and
    the current strategy.

    A developer can also choose to review a case that needs reviewing.
    """
    my_dev_pile = simpy.Store(env, DEVELOPER_CAPACITY)
    my_review_pile = simpy.Store(env, DEVELOPER_REVIEW_CAPACITY)
    my_merge_pile = simpy.Store(env)  # Devs always have merge capacity for their own cases

    while True:
        # -----------------------
        # Populate work queues
        # -----------------------
        # Only try to fetch a new case if this developer has spare capacity
        if len(my_dev_pile.items) < DEVELOPER_CAPACITY and len(work_piles[WORK_DEV].items) > 0:

            # Get a case off the backlog
            logging.debug('%s requesting a case at %s' % (name, env.now))
            case = yield work_piles[WORK_DEV].get()

            # If there was a case to get, add it to my pile
            if case:
                logging.debug('%s got case %s at %s' % (name, case, env.now))
                my_dev_pile.put(case)

        # Only try fetching a review item if we have spare review capacity
        reviewable_items = [x for x in work_piles[WORK_REVIEW].items if developer_of_work_item(x) != name]
        if len(my_review_pile.items) < DEVELOPER_REVIEW_CAPACITY and len(reviewable_items) > 0:
            # A review work item might be part of a DevWorkflow that restricts the review item to not be done by
            # the same dev that did the dev work in the same workflow.
            logging.debug('%s requesting a case to review at %s' % (name, env.now))
            work = yield work_piles[WORK_REVIEW].get(lambda x: developer_of_work_item(x) != name)

            if work:
                logging.debug('%s got %s at %s' % (name, work, env.now))
                my_review_pile.put(work)

        # Fetch a merge item if there's one to fetch
        mergeable_items = [x for x in work_piles[WORK_MERGE].items if developer_of_work_item(x) == name]
        if len(mergeable_items) > 0:
            # A merge work item must be done by the dev associated with the workflow
            logging.debug('%s requesting a case to merge at %s' % (name, env.now))
            work = yield work_piles[WORK_MERGE].get(lambda x: developer_of_work_item(x) == name)

            if work:
                logging.debug('%s got %s at %s' % (name, work, env.now))
                my_merge_pile.put(work)

        # Either no more items to work on or review since we started or we haven't started yet
        if len(my_dev_pile.items) == 0 and len(my_review_pile.items) == 0 and len(my_merge_pile.items) == 0:
            #logging.debug('%s has nothing to work on at %s' % (name, env.now))
            yield env.timeout(1)
            continue

        # -----------------------
        # Choose what to work on (dev or review work).
        # -----------------------
        # Current strategy: If there's a work item to merge or review, they take priority over your dev work so take
        # the next item to review
        if len(my_merge_pile.items) > 0:
            item_to_work_on = yield my_merge_pile.get()
        elif len(my_review_pile.items) > 0:
            item_to_work_on = yield my_review_pile.get()
        else:
            # Nothing to review or merge so do some dev work
            if strategy == RANDOM:
                # Strategy #1: Choose randomly
                np.random.seed(datetime.now().microsecond)
                item_to_work_on = my_dev_pile.items[np.random.randint(len(my_dev_pile.items))]
            elif strategy in (BIGGEST_FIRST, SMALLEST_FIRST):
                if strategy == SMALLEST_FIRST:
                    item_index = 0
                else:
                    item_index = -1
                item_to_work_on = sorted(my_dev_pile.items, key=lambda x: x.size())[item_index]
            elif strategy in (HIGHEST_COST_FIRST, HIGHEST_VALUE_FIRST):
                # Strategy #2: Work on the highest cost case first (max negative value)
                # Strategy #3: Work on the highest value case first (max positive value)
                if strategy == HIGHEST_COST_FIRST:
                    item_index = 0
                else:
                    item_index = -1
                item_to_work_on = sorted(my_dev_pile.items, key=lambda x: x.case.value)[item_index]
            elif strategy == HIGHEST_VALUE_TO_COST_RATIO_FIRST:
                item_to_work_on = sorted(my_dev_pile.items, key=lambda x: x.case.value/x.size())[-1]
            else:
                raise Exception('Invalid dev strategy: %s' % strategy)

        # -----------------------
        # Work on the item. If it's dev work we can do it a bit at a time, but a review work must be completed
        # before moving on to the next item.
        # -----------------------
        logging.debug('%s working on %s at %s' % (name, item_to_work_on, env.now))

        if item_to_work_on.name == WORK_DEV:
            if not item_to_work_on.is_done():
                # do some work
                item_to_work_on.do_work(by=name)
                logging.debug('%s worked on %s at %s' % (name, item_to_work_on, env.now))

                if item_to_work_on.is_done():
                    # Remove it from the pile
                    my_dev_pile.items.remove(item_to_work_on)

        elif item_to_work_on.name in (WORK_REVIEW, WORK_MERGE):
            # Work on review or merge work until it's done
            yield env.timeout(item_to_work_on.size())
            item_to_work_on.finish(by=name)
            logging.info('%s finished %s at %s' % (name, item_to_work_on, env.now))

        else:
            raise Exception('Invalid work type for dev: %s' % item_to_work_on.name)

        # Do this every day
        yield env.timeout(1)


def qa(name, env, current_release=None):
    """
    A QA takes a case off the QA pile, works on it for a period of time proportional to its size then marks it as
    finished. They can only work on one at a time.
    """
    while True:
        # If a release is in test, we're regression testing it so don't work on cases
        if not current_release is None and current_release.in_test:
            logging.debug('%s Doing regression testing at %s' % (name, env.now))
            yield env.timeout(1)
            continue

        # Get a work item off the QA backlog
        logging.debug('%s requesting work at %s' % (name, env.now))
        work = yield work_piles[WORK_QA].get()
        logging.debug('%s got work %s at %s' % (name, work, env.now))

        size = work.size()
        yield env.timeout(size)
        work.finish(by=name)
        logging.info('%s finished %s at %s' % (name, work, env.now))


def releaser(name, env, current_release=None, releases=None):
    """
    A Releaser takes cases off the done pile and assigns them to a release. QA people can then do regression testing release
    after which all the cases in the release can be marked as released.
    """
    while True:
        # Get a case off the Done pile and add it to a release
        logging.debug('%s requesting a case at %s' % (name, env.now))
        case = yield done_pile.get()
        logging.debug('%s got %s at %s' % (name, case, env.now))

        # If the release has enough cases on it or this is the balance of the releasable cases
        # mark it as released, mark all associated cases as released and update value
        # todo: or add a time limit
        if current_release is None:
            current_release = Release(env)

        logging.debug('case: %s, released_pile: %s, current_release: %s at %s' % (case.name, len(released_pile.items), len(current_release.cases), env.now))

        # Just add the case to the current release
        current_release.add_case(case)

        # Do a release if necessary
        if len(current_release.cases) >= MAX_RELEASE_CASES or (len(current_release.cases) + len(released_pile.items)) == len(standard_cases):

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


def sourcer(name, env):
    """
    The job of the sourcer is just to generate new cases by adding them from the hidden backlog periodically.
    """
    while True:
        # Get a case off the Source backlog
        if len(work_piles[WORK_ANALYSIS].items) <= ANALYSIS_PILE_MIN:
            logging.debug('%s fetching a source case at %s' % (name, env.now))
            case = yield source_pile.get()

            # Wait the specified time before adding it to the dispatch pile (for work allocation) and the progress pile
            # (for tracking)
            yield env.timeout(NEW_CASE_INTERVAL)

            # Add work items for the new case to the dispatch pile
            dispatch_new_case_work(case)
            logging.info('%s dispatching %s at %s' % (name, case, env.now))
        else:
            yield env.timeout(1)


def dispatch_new_case_work(case):
    """
    Initialises work queues with work items for a new case
    :param case:
    :return:
    """
    # Track the case on the progress pile
    progress_pile.put(case)

    # Add the cases first work items to the dispatch pile
    items = case.workflow.get_next_step()
    if items is not None:
        for item in items:
            dispatch_work_item(item)


def dispatch_work_item(item):
    """
    Add a work item to the relevant queue depending on whether it is a Workflow or Work item
    :param items:
    :return:
    """
    if item is not None:
        if type(item) is Workflow:
            for work in item.work_items:
                dispatch_pile.append(work)
        elif type(item) is Work:
            assign_work_to_pile(item)
        else:
            raise Exception("Incorrect work item type: %s" % type(item))


def dispatcher(name, env):
    """
    This actor distributes cases to different piles when the current assignee has relinquished it.
    This way an individual actor doesn't need to know what happens to the case next after they've finished with it
    :param name:
    :param env:
    :return: void
    """
    while True:
        # Get all work items off the dispatch backlog, work out what needs to happen the associated case next and
        # add the relevant items to the relevant piles.
        for item in dispatch_pile:
            dispatch_work_item(item)
            logging.debug('%s dispatched work for work item %s at %s' % (name, item, env.now))

        # Clear the dispatch pile
        dispatch_pile.clear()

        # Do this once a day
        yield env.timeout(1)


def case_done_listener(case=None):
    # Move the case to the done_pile pile
    done_pile.put(case)


pub.subscribe(case_done_listener, 'case_is_done')


def assign_work_to_pile(work):
    """
    Assign a work item to the relevant pile depending on the type of the work
    :param work:
    :return:
    """
    if work.name in work_piles.keys():
        logging.debug('Assigning work item %s to pile %s at %s' % (work, work.name, env.now))
        work_piles[work.name].put(work)


def plot_ecdf(data, color='black'):
    _ = plt.plot(np.sort(data), np.arange(1, len(data) + 1) / len(data), marker='.', linestyle='none', alpha=0.01, color=color)
    return _


# Create a repeatable backlog of cases that have binomially distributed sizes and normally distributed values
logging.info('Creating standard cases')
np.random.seed(RANDOM_SEED)

standard_cases = []
sizes = np.random.gamma(10, size=NUM_SOURCE_CASES).astype(np.int64)
values = (np.random.normal(0, 1, size=NUM_SOURCE_CASES)) * 10
for idx, size in enumerate(sizes):
    standard_cases.append(Case(size=max(size, 1), value=values[idx], name='%d' % idx))  # value = size

# Do multiple executions of this model where behaviour of agents varies
runs = []
releases = []
cases = []
current_release = None  # a global resource, can only be one release active at a time
sourcers = []
dispatchers = []
finisher = None

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
    merge_size = []
    cycle_times = []
    total_values = []

    # Choose a random strategy for each run
    strategy = np.random.randint(6)
    ba_strategy = np.random.randint(2)
    dev_review_choice_strategy = np.random.randint(3)

    env = simpy.Environment()

    # The dispatch pile is just a list, as it gets processed entirely each day
    dispatch_pile = []

    # Create standard piles
    source_pile = simpy.Store(env, len(sizes))
    done_pile = simpy.Store(env)
    released_pile = simpy.Store(env)
    progress_pile = simpy.Store(env)

    # Create named piles for specific work types, i.e., work that needs to be done to get the case to the Done pile
    work_piles = {
        WORK_ANALYSIS: simpy.Store(env, len(sizes)),
        WORK_QA: simpy.Store(env, MAX_QA_PILE),
        WORK_DEV: simpy.Store(env, MAX_DEV_PILE),
        WORK_REVIEW: simpy.FilterStore(env, MAX_REVIEW_PILE),
        WORK_MERGE: simpy.FilterStore(env)
    }

    # Create a Run to store params
    run = Run(params={
            'strategy': strategy,
            'ba_strategy': ba_strategy,
            'dev_review_choice_strategy': dev_review_choice_strategy
    })

    # Create a pile of all remaining source cases
    for case in standard_cases[NUM_INITIAL_CASES:len(standard_cases)]:
        _case = copy.deepcopy(case)
        _case.set_env(env)
        source_pile.put(_case)

    # Create a pile of cases to get the system to a steady state. We will start by putting the first workflow step
    # for each case on the dispatcher's pile and let them allocate work based on the workflow step properties.
    for case in standard_cases[:NUM_INITIAL_CASES]:
        _case = copy.deepcopy(case)
        _case.set_env(env)
        dispatch_new_case_work(_case)

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

    # Create a Dispatcher
    dispatchers = [env.process(dispatcher('Dispatcher 0', env))]

    # Create a case Sourcer
    sourcers = [env.process(sourcer('Sourcer 0', env))]

    # add monitoring, which will also terminate the sim on completion of all work
    finisher = simpy.Event(env)
    env.process(monitor(env, finisher))

    # Execute!
    env.run(until=finisher)

    # Store cycle time values
    for case in released_pile.items:
        cycle_times.append(case.cycle_time())

    # Store stats for this run
    run.data = {
        WORK_ANALYSIS: analysis_size,
        WORK_DEV: dev_size,
        WORK_REVIEW: review_size,
        WORK_QA: qa_size,
        WORK_MERGE: merge_size,
        'done': done_size,
        'release': release_size,
        'cycle_time': cycle_times,
        'total_value': total_values
    }

    runs.append(run)

#sys.exit(0)

# Counts in each state
for i, data in enumerate([x.run_data(WORK_ANALYSIS) for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='blue')
for i, data in enumerate([x.run_data(WORK_DEV) for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='red')
for i, data in enumerate([x.run_data(WORK_QA) for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='green')
for i, data in enumerate([x.run_data(WORK_REVIEW) for x in runs]):
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