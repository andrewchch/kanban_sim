from models.work import Work
from functools import reduce
from pubsub import pub


class Workflow:
    """
    A workflow containing some work items (or potentially other workflows) to do and how to manage them, e.g., the
    assignee can do them or must delegate them to someone else. Workflow steps can contain other Workflow steps so an
    entire Workflow can be defined by one WorkflowStep
    """
    def __init__(self, work_items=[], delegate=False, sequential=True, case=None, workflow=None):
        assert (work_items is not None and type(work_items) == list and len(work_items) > 0),\
            "Need to specify some work items to do"

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
        self.end_time = None
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
                    pub.sendMessage('dispatch_work', work=item)

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
