"""Placeholder for the future MELD scheduler implementation."""
from pathlib import Path

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ModelManager import Contract
from utils.config import SCHEDULE_DIR

class SchedulerService:

    def __init__(self, execution_service: "ExecutionService"):
        self.execution_service = execution_service
        self._create_schedules_folder()

        self._scheduler = BackgroundScheduler(
            jobstores={
                "default": SQLAlchemyJobStore(
                    url=f"sqlite:///{SCHEDULE_DIR}/scheduler.db"
                )
            }
        )

    def _create_schedules_folder(self) -> Path:
        SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
        return SCHEDULE_DIR

    def start(self):
        self._scheduler.start()

    def validate_schedule(self, schedule) -> None:
        if schedule is not None:
            CronTrigger.from_crontab(schedule.expression, schedule.timezone)

    def add_job(self, contract: Contract):
        schedule = contract.schedule
        if schedule is None:
            return

        self.validate_schedule(schedule)
        self._scheduler.add_job(self.execution_service.start_execution,
                                CronTrigger.from_crontab(schedule.expression, schedule.timezone),
                                args=[contract.id],
                                id=contract.id,
                                jobstore="default",
                                replace_existing=True,
                                max_instances=1,
                                )

    def remove_job(self, contract: Contract):
        self._scheduler.remove_job(contract.id,
                                   jobstore="default")
