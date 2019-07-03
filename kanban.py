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
8. Agents can learn over time from overall metrics, or from other agents (other agents broadcast their states). If we mimic an actual
Kanban process then all agents have perfect information of the workloads of other agents and can alter their behaviour accordingly. If
(as in reality) agents' (bounded) rationality is to maximise their own business, then we should be able to see the effect of this.
9. Monitor blocking and starving metrics for each queue.
"""
import sys
import simpy
import numpy as np
import matplotlib.pyplot as plt
import logging
import copy

from tqdm import tqdm
from pubsub import pub
from models import Run, Case, Workflow, Work
from agents import BA, QA, Releaser, Developer
from workflows import WorkTypes as wt
from workflows.standard import StandardWorkflowFactory

RANDOM_SEED = 42
SIM_TIME = 1000  # Simulation time in days
NUM_DEVELOPERS = 8
NUM_QA = 2
NUM_BA = 1
NUM_RUNS = 200
#NUM_RUNS = 1
MAX_QA_PILE = 10
MAX_DEV_PILE = 20
MAX_REVIEW_PILE = 20

NUM_SOURCE_CASES = 300
NUM_INITIAL_CASES = 100
#NUM_SOURCE_CASES = 20
#NUM_INITIAL_CASES = 10

NEW_CASE_INTERVAL = 5
ANALYSIS_PILE_MIN = 5

logging.basicConfig(level=logging.ERROR,
                    format='%(message)s',)


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
        analysis_size.append(len(work_piles[wt.WORK_ANALYSIS].items))
        dev_size.append(len(work_piles[wt.WORK_DEV].items))
        dev_put_queue_size.append(len(work_piles[wt.WORK_DEV].put_queue))
        qa_size.append(len(work_piles[wt.WORK_QA].items))
        merge_size.append(len(work_piles[wt.WORK_MERGE].items))
        done_size.append(len(done_pile.items))
        release_size.append(len(released_pile.items))

        yield env.timeout(1)

def sourcer(name, env):
    """
    The job of the sourcer is just to generate new cases by adding them from the hidden backlog periodically.
    """
    while True:
        # Get a case off the Source backlog
        if len(work_piles[wt.WORK_ANALYSIS].items) <= ANALYSIS_PILE_MIN:
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
    :param item:
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
    assert case is not None, "case_done_listener: No case provided"
    done_pile.put(case)


def dispatch_work_listener(work=None):
    # Move the case to the done_pile pile
    assert work is not None, "dispatch_work_listener: No work provided"
    dispatch_pile.append(work)


pub.subscribe(case_done_listener, 'case_is_done')
pub.subscribe(dispatch_work_listener, 'dispatch_work')


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
    standard_cases.append(Case(size=max(size, 1), value=values[idx], name='%d' % idx,
                               workflow=StandardWorkflowFactory.make_workflow(dev_size=size)))  # value = size

# Do multiple executions of this model where behaviour of models varies
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
    ba_strategy = BA.random_strategy()
    qa_strategy = QA.random_strategy()
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
        wt.WORK_ANALYSIS: simpy.Store(env, len(sizes)),
        wt.WORK_QA: simpy.Store(env, MAX_QA_PILE),
        wt.WORK_DEV: simpy.Store(env, MAX_DEV_PILE),
        wt.WORK_REVIEW: simpy.FilterStore(env, MAX_REVIEW_PILE),
        wt.WORK_MERGE: simpy.FilterStore(env)
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
        developers.append(env.process(Developer('Developer %d' % i,
                                                env,
                                                strategy=strategy,
                                                dev_pile=work_piles[wt.WORK_DEV],
                                                review_pile=work_piles[wt.WORK_REVIEW],
                                                merge_pile=work_piles[wt.WORK_MERGE]
                                                ).run()))

    # Create some QAs
    qas = []
    for i in range(NUM_QA):
        qas.append(env.process(QA('QA %d' % i, env, qa_strategy, work_piles[wt.WORK_QA], current_release).run()))

    # Create some BAs
    bas = []
    for i in range(NUM_BA):
        bas.append(env.process(BA('BA %d' % i, env, ba_strategy, work_piles[wt.WORK_ANALYSIS]).run()))

    # Create a Releaser
    releasers = [env.process(Releaser('Releaser 0', env, current_release, done_pile, releases, released_pile,
                                      standard_cases).run())]

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
        wt.WORK_ANALYSIS: analysis_size,
        wt.WORK_DEV: dev_size,
        wt.WORK_REVIEW: review_size,
        wt.WORK_QA: qa_size,
        wt.WORK_MERGE: merge_size,
        'done': done_size,
        'release': release_size,
        'cycle_time': cycle_times,
        'total_value': total_values
    }

    runs.append(run)

#sys.exit(0)

# Counts in each state
for i, data in enumerate([x.run_data(wt.WORK_ANALYSIS) for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='blue')
for i, data in enumerate([x.run_data(wt.WORK_DEV) for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='red')
for i, data in enumerate([x.run_data(wt.WORK_QA) for x in runs]):
    _ = plt.plot(data, alpha=0.1, color='green')
for i, data in enumerate([x.run_data(wt.WORK_REVIEW) for x in runs]):
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