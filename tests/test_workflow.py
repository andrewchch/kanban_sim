import unittest
import logging

from kanbansim.workflows.standard import StandardWorkflowFactory
from kanbansim.models import Work, Workflow
import simpy

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',)


def finish_all_item_work(item):
    if item is not None:
        if issubclass(type(item), Workflow):
            logging.info('Finishing %s...' % item)
            while True:
                items = item.get_next_step()
                if items is None:
                    logging.info('No more items for workflow %s' % item)
                    break
                for _item in tuple(items):
                    logging.info('Finishing item %s...' % _item)
                    finish_all_item_work(_item)

        elif type(item) is Work:
            logging.info('Finished %s work' % item)
            item.finish(by='unittest')


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = StandardWorkflowFactory.make_workflow(dev_size=10)
        self.env = simpy.Environment()
        self.workflow.set_env(self.env)

    def test_create_workflow(self):
        self.assertEqual(self.workflow.is_done(), False)
        self.assertEqual(len(self.workflow.work_items), 2)

    def test_progress_workflow(self):
        self.assertEqual(self.workflow.is_done(), False)

    def test_simple_workflow_is_done(self):
        """
        Add a single work item, finish it and confirm that the workflow is done.
        :return:
        """
        work = Work(size=1, workflow=self.workflow, env=self.env)
        self.workflow.work_items = [work]
        work.finish(by='me')
        self.assertEqual(self.workflow.is_done(), True)

    def test_multistep_workflow_with_finished_items_is_done(self):
        """
        Add a workflow work item with its own work items, finish them and confirm that the top level workflow is done.
        :return:
        """
        child_workflow = Workflow(
                work_items=[
                    Work(size=1, workflow=self.workflow, env=self.env),
                    Work(size=1, workflow=self.workflow, env=self.env),
                    Work(size=1, workflow=self.workflow, env=self.env)
                ],
                workflow=self.workflow
            )

        self.workflow.work_items = [child_workflow]

        self.assertEqual(self.workflow.is_done(), False)

        for item in child_workflow.work_items:
            item.finish(by='me')

        self.assertEqual(self.workflow.is_done(), True)

    def test_standard_workflow_to_completion(self):
        """
        Process all steps of a standard workflow until it's done.
        :return:
        """
        self.assertEqual(False, self.workflow.is_done())
        finish_all_item_work(self.workflow)
        self.assertEqual(True, self.workflow.is_done())


