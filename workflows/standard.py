from models import Work, Workflow, DevWorkflow
from workflows import WorkTypes


class StandardWorkflowFactory:

    # Work names
    ANALYSIS_WORK_PER_CASE = 2
    QA_WORK_PER_CASE = 4
    MERGE_WORK_PER_CASE = 1
    REVIEW_WORK_RATIO = 0.5

    @classmethod
    def make_workflow(cls, dev_size=0):
        assert dev_size > 0, "Dev task size must be greater than 0"
        return Workflow(
                work_items=[
                    Work(name=WorkTypes.WORK_ANALYSIS, size=cls.ANALYSIS_WORK_PER_CASE),
                    DevWorkflow(work_items=[
                        Work(name=WorkTypes.WORK_DEV, size=dev_size),
                        Workflow(
                            work_items=[
                                Work(name=WorkTypes.WORK_REVIEW, size=int(dev_size * cls.REVIEW_WORK_RATIO)),
                                Work(name=WorkTypes.WORK_REVIEW, size=int(dev_size * cls.REVIEW_WORK_RATIO))
                            ],
                            sequential=False
                        ),
                        Work(name=WorkTypes.WORK_QA, size=cls.QA_WORK_PER_CASE),
                        Work(name=WorkTypes.WORK_MERGE, size=cls.MERGE_WORK_PER_CASE)
                    ]),
                ],
                sequential=True
            )
