import logging


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
