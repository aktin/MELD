import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import test_support  # noqa: F401
except ModuleNotFoundError:
    from MELD.tests import test_support  # noqa: F401
from apscheduler.jobstores.base import JobLookupError

from ExecutionScheduler import SchedulerService
from ModelManager import Contract


CONTRACT_DATA = test_support.contract_data()


class SchedulerServiceTest(unittest.TestCase):
    def contract(self, schedule=None, name="example-contract"):
        data = {
            **CONTRACT_DATA,
            "contract": {
                **CONTRACT_DATA["contract"],
                "name": name,
                "version": "1-0-0",
            },
        }
        if schedule is not None:
            data["schedule"] = schedule
        return Contract.from_dict(data)

    def service(self, directory, execution_service=None):
        with patch("ExecutionScheduler.scheduler.SCHEDULE_DIR", Path(directory)):
            return SchedulerService(execution_service or MagicMock())

    def test_constructor_creates_schedule_directory_and_configures_persistent_store(self):
        with tempfile.TemporaryDirectory() as directory:
            execution_service = MagicMock()
            service = self.service(Path(directory) / "schedules", execution_service)

            self.assertTrue(Path(directory, "schedules").is_dir())
            self.assertIs(service.execution_service, execution_service)
            self.assertEqual(
                service._scheduler._jobstores["default"].engine.url.database,
                str(Path(directory, "schedules", "scheduler.db")),
            )
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_start_delegates_to_scheduler(self):
        execution_service = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            with patch("ExecutionScheduler.scheduler.BackgroundScheduler") as scheduler_type:
                service = self.service(directory, execution_service)
                service.start()

            scheduler_type.return_value.start.assert_called_once_with()

    def test_validate_schedule_accepts_none_and_valid_cron(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            service.validate_schedule(None)
            service.validate_schedule(self.contract({"expression": "* * * * *"}).schedule)
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_validate_schedule_rejects_invalid_cron(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            with self.assertRaises(ValueError):
                service.validate_schedule(
                    type("Schedule", (), {"expression": "invalid", "timezone": "Europe/Berlin"})()
                )
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_add_job_without_schedule_does_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            with patch.object(service._scheduler, "add_job") as add_job:
                service.add_job(self.contract())
            add_job.assert_not_called()
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_add_job_registers_execution_callback_and_limits_instances(self):
        execution_service = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, execution_service)
            contract = self.contract({"expression": "* * * * *"})
            with patch.object(service._scheduler, "add_job") as add_job:
                service.add_job(contract)

            args, kwargs = add_job.call_args
            self.assertIs(args[0], execution_service.start_execution)
            self.assertEqual(kwargs["args"], [contract.id])
            self.assertEqual(kwargs["id"], contract.id)
            self.assertEqual(kwargs["jobstore"], "default")
            self.assertTrue(kwargs["replace_existing"])
            self.assertEqual(kwargs["max_instances"], 1)
            self.assertIn("minute", {field.name for field in args[1].fields})
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_add_job_validates_before_registering(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            contract = self.contract({"expression": "* * * * *"})
            with patch.object(service, "validate_schedule", side_effect=ValueError("bad")):
                with patch.object(service._scheduler, "add_job") as add_job:
                    with self.assertRaisesRegex(ValueError, "bad"):
                        service.add_job(contract)
            add_job.assert_not_called()
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_remove_job_delegates_contract_id_and_jobstore(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            contract = self.contract({"expression": "* * * * *"})
            with patch.object(service._scheduler, "remove_job") as remove_job:
                service.remove_job(contract)
            remove_job.assert_called_once_with(contract.id, jobstore="default")
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_remove_missing_job_propagates_lookup_error(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            with self.assertRaises(JobLookupError):
                service.remove_job(self.contract({"expression": "* * * * *"}))
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)

    def test_job_persists_replaces_and_can_be_removed_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            schedule_dir = Path(directory) / "schedules"
            contract = self.contract({"expression": "* * * * *"})
            first = self.service(schedule_dir)
            first.add_job(contract)
            first.start()
            first._scheduler.shutdown(wait=False)

            second = self.service(schedule_dir)
            second.start()
            self.assertIsNotNone(second._scheduler.get_job(contract.id, jobstore="default"))
            second.add_job(contract)
            self.assertEqual(len(second._scheduler.get_jobs()), 1)
            second.remove_job(contract)
            self.assertIsNone(second._scheduler.get_job(contract.id, jobstore="default"))
            second._scheduler.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
