from models.workflow import Workflow, DevWorkflow
from models.work import Work


class Case:
    """
    A task to be done dictated by a Workflow. A case will have a primary assignee by virtue of being
    on that actor's backlog, but individual work items (e.g., review work) can be picked up by other actors.
    A case is primarily used to track overall metrics, e.g., cycle time, current value
    """
    issuetype = 'Bug'  # todo: implement case types
    index = 0

    def __init__(self, size=0, value=0, env=None, name=None, workflow=None):
        assert size is not None, "Size must be greater than zero"
        assert value is not None and value != 0, "Value must be non-zero"
        assert workflow is not None, "Must provide a workflow"
        self.workflow = workflow
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
