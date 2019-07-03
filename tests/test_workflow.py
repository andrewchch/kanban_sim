import unittest

from workflows.standard import *


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.workflow = make_workflow(dev_size=10)

    def test_create_workflow(self):
        self.assertEqual(self.workflow.is_done(), False)
        self.assertEqual(len(self.workflow.work_items), 2)

    def test_progress_workflow(self):
        self.assertEqual(self.workflow.is_done(), False)
