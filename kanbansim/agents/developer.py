import logging
import simpy
import numpy as np

from agents.agent import Agent
from datetime import datetime
from models import Work, DevWorkflow
from workflows import WorkTypes as wt


def developer_of_work_item(item):
    """
    Get the developer of the dev work associated with this review or merge item
    :param item:
    :return:
    """
    assert type(item) is Work, "Not a work item"

    # Get the grandparent workflow
    dev_workflow = None

    if wt.WORK_REVIEW == item.name:
        review_workflow = item.workflow
        if review_workflow is not None:
            dev_workflow = review_workflow.workflow
    elif item.name == wt.WORK_MERGE:
        dev_workflow = item.workflow

    if dev_workflow is not None and type(dev_workflow) is DevWorkflow:
        dev_work = dev_workflow.work_items[0]
        return dev_work.work_done_by

    return None


class DeveloperConfig:
    DEVELOPER_REVIEW_CAPACITY: int = 2
    DEVELOPER_CAPACITY: int = 4

    def __init__(self, developer_review_capacity=DEVELOPER_REVIEW_CAPACITY, developer_capacity=DEVELOPER_CAPACITY):
        assert developer_review_capacity > 0, "developer_review_capacity must be > 0"
        assert developer_capacity > 0, "developer_capacity must be > 0"
        self.developer_review_capacity = developer_review_capacity
        self.developer_capacity = developer_capacity


class Developer(Agent):

    HIGHEST_COST_FIRST = 0
    HIGHEST_VALUE_FIRST = 1
    BIGGEST_FIRST = 2
    SMALLEST_FIRST = 3
    HIGHEST_VALUE_TO_COST_RATIO_FIRST = 4
    RANDOM = 5

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

    strategies = [
        'Highest Cost First',
        'Highest Value First',
        'Biggest First',
        'Smallest First',
        'Highest Value-to-Cost Ratio First',
        'Random'
    ]

    _review_strategies = [
        'Reviews First',
        'Random',
        'Development Work First'
    ]

    @classmethod
    def random_review_strategy(cls):
        return np.random.choice(len(cls.strategies))

    def __init__(self, name, env, strategy, dev_pile, review_pile, merge_pile, conf: DeveloperConfig = None):
        self.name = name
        self.env = env
        self.strategy = strategy
        self.dev_pile = dev_pile
        self.review_pile = review_pile
        self.merge_pile = merge_pile
        self.conf = conf or DeveloperConfig()

    def run(self):
        """A developer takes a work item off the backlog, works on it for a period of time proportional to its size,
        then marks it as done.
    
        Developers can have multiple cases on the go (up to LIMIT), and each day they can choose to work on one case or
        another until it is done.
    
        The Developer has knowledge of the value of a case in order to be able to prioritise cases according to value\
        and the current strategy.
    
        A developer can also choose to review a case that needs reviewing.
        """
        my_dev_pile = simpy.Store(self.env, self.conf.developer_capacity)
        my_review_pile = simpy.Store(self.env, self.conf.developer_review_capacity)
        my_merge_pile = simpy.Store(self.env)  # Devs always have merge capacity for their own cases
    
        while True:
            # -----------------------
            # Populate work queues
            # -----------------------
            # Only try to fetch a new case if this developer has spare capacity
            if len(my_dev_pile.items) < self.conf.developer_capacity and len(self.dev_pile.items) > 0:
    
                # Get a work item off the backlog
                logging.debug('%s requesting a case at %s' % (self.name, self.env.now))
                work = yield self.dev_pile.get()
    
                # If there was a case to get, add it to my pile
                if work:
                    logging.debug('%s got case %s at %s' % (self.name, work, self.env.now))
                    my_dev_pile.put(work)
    
            # Only try fetching a review item if we have spare review capacity
            reviewable_items = [x for x in self.review_pile.items if developer_of_work_item(x) != self.name]
            if len(my_review_pile.items) < self.conf.developer_review_capacity and len(reviewable_items) > 0:
                # A review work item might be part of a DevWorkflow that restricts the review item to not be done by
                # the same dev that did the dev work in the same workflow.
                logging.debug('%s requesting a case to review at %s' % (self.name, self.env.now))
                work = yield self.review_pile.get(lambda x: developer_of_work_item(x) != self.name)
    
                if work:
                    logging.debug('%s got %s at %s' % (self.name, work, self.env.now))
                    my_review_pile.put(work)
    
            # Fetch a merge item if there's one to fetch
            mergeable_items = [x for x in self.merge_pile.items if developer_of_work_item(x) == self.name]
            if len(mergeable_items) > 0:
                # A merge work item must be done by the dev associated with the workflow
                logging.debug('%s requesting a case to merge at %s' % (self.name, self.env.now))
                work = yield self.merge_pile.get(lambda x: developer_of_work_item(x) == self.name)
    
                if work:
                    logging.debug('%s got %s at %s' % (self.name, work, self.env.now))
                    my_merge_pile.put(work)
    
            # Either no more items to work on or review since we started or we haven't started yet
            if len(my_dev_pile.items) == 0 and len(my_review_pile.items) == 0 and len(my_merge_pile.items) == 0:
                # logging.debug('%s has nothing to work on at %s' % (name, env.now))
                yield self.env.timeout(1)
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
                if self.strategy == self.RANDOM:
                    # Strategy #1: Choose randomly
                    np.random.seed(datetime.now().microsecond)
                    item_to_work_on = my_dev_pile.items[np.random.randint(len(my_dev_pile.items))]
                elif self.strategy in (self.BIGGEST_FIRST, self.SMALLEST_FIRST):
                    if self.strategy == self.SMALLEST_FIRST:
                        item_index = 0
                    else:
                        item_index = -1
                    item_to_work_on = sorted(my_dev_pile.items, key=lambda x: x.size())[item_index]
                elif self.strategy in (self.HIGHEST_COST_FIRST, self.HIGHEST_VALUE_FIRST):
                    # Strategy #2: Work on the highest cost case first (max negative value)
                    # Strategy #3: Work on the highest value case first (max positive value)
                    if self.strategy == self.HIGHEST_COST_FIRST:
                        item_index = 0
                    else:
                        item_index = -1
                    item_to_work_on = sorted(my_dev_pile.items, key=lambda x: x.case.value)[item_index]
                elif self.strategy == self.HIGHEST_VALUE_TO_COST_RATIO_FIRST:
                    item_to_work_on = sorted(my_dev_pile.items, key=lambda x: x.case.value/x.size())[-1]
                else:
                    raise Exception('Invalid dev strategy: %s' % self.strategy)
    
            # -----------------------
            # Work on the item. If it's dev work we can do it a bit at a time, but a review work must be completed
            # before moving on to the next item.
            # -----------------------
            logging.debug('%s working on %s at %s' % (self.name, item_to_work_on, self.env.now))
    
            if item_to_work_on.name == wt.WORK_DEV:
                if not item_to_work_on.is_done():
                    # do some work
                    item_to_work_on.do_work(by=self.name)
                    logging.debug('%s worked on %s at %s' % (self.name, item_to_work_on, self.env.now))
    
                    if item_to_work_on.is_done():
                        # Remove it from the pile
                        my_dev_pile.items.remove(item_to_work_on)
    
            elif item_to_work_on.name in (wt.WORK_REVIEW, wt.WORK_MERGE):
                # Work on review or merge work until it's done
                yield self.env.timeout(item_to_work_on.size())
                item_to_work_on.finish(by=self.name)
                logging.info('%s finished %s at %s' % (self.name, item_to_work_on, self.env.now))
    
            else:
                raise Exception('Invalid work type for dev: %s' % item_to_work_on.name)
    
            # Do this every day
            yield self.env.timeout(1)
