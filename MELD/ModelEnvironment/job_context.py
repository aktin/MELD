import datetime
import json
import os.path
from enum import Enum

from docker.utils import kwargs_from_env

from ModelManager import config_loader
from meld_logger import setup_logger


class JobStatus(Enum):
    QUERY_FINISHED = "QUERY_FINISHED"
    START_QUERY = "START_QUERY"
    PENDING = "PENDING"
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"
    DESTROYED = "DESTROYED"
    PULLING_IMAGE = "PULLING_IMAGE"
    IMAGE_PULLED = "IMAGE_PULLED"


class JobContext:
    def __init__(self, contract_path: str):
        self.status = None
        self.container_status = None

        self.contract_path = contract_path
        self.contract = config_loader.load_contract(contract_path)

        self.job_id = self._create_job_id()

        # set up folder structure
        self._job_folder = self._create_job_folder()
        self.input_data_path = self._create_input_folder()
        self.output_data_path = self._create_output_folder()
        self.status_path = self._create_status_folder()
        self.logs_path = self._create_log_folder()
        self.query_path = self._get_query_path()

        self.logger = setup_logger(f"meld.job{self.job_id}", logs_path=self.logs_path, )
        self.log_event("Job created", JobStatus.PENDING)


    def _get_query_path(self):
        query_path = os.path.join("/resources", self.contract["input_schema"]["query"]["path"])
        if not os.path.exists(query_path):
            raise Exception(f"Query file for job {self.job_id} does not exist")
        return query_path

    def _create_input_folder(self):
        input_path = os.path.join(self._job_folder, "input")

        if not os.path.exists(input_path):
            os.makedirs(input_path)
        else:
            raise Exception(f"Input folder for job {self.job_id} already exists")

        return input_path

    def _create_output_folder(self):
        input_path = os.path.join(self._job_folder, "output")

        if not os.path.exists(input_path):
            os.makedirs(input_path)
        else:
            raise Exception(f"Output folder for job {self.job_id} already exists")

        return input_path

    def _create_job_folder(self):
        job_folder = os.path.join("/jobs", self.job_id)

        if not os.path.exists(job_folder):
            os.makedirs(job_folder)
        else:
            raise Exception(f"Job folder for job {self.job_id} already exists")

        return job_folder

    def _create_job_id(self):
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    def set_status(self, status: JobStatus):
        self.status = status
        with open(os.path.join(self.status_path, "status.json"), "w") as f:
            json.dump({
                "status": status.value,
                "lastUpdated": datetime.datetime.now().isoformat(),
                "jobId": self.job_id
            }, f, sort_keys=True, indent=4)

    def log_event(self, message: str, event: JobStatus, **kwargs):
        self.logger.info(message)
        self.set_status(event)
        with open(os.path.join(self.status_path, "events.jsonl"), "a") as f:
            f.write(json.dumps(
                {"message": message, "event": event.value, "timestamp": datetime.datetime.now().isoformat(), **kwargs},
                sort_keys=True) + "\n")

    def _create_status_folder(self):
        status_path = os.path.join(self._job_folder, "status")

        if not os.path.exists(status_path):
            os.makedirs(status_path)
        else:
            raise Exception(f"Status folder for job {self.job_id} already exists")

        return status_path

    def _create_log_folder(self):
        log_path = os.path.join(self._job_folder, "logs")

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            raise Exception(f"Log folder for job {self.job_id} already exists")

        return log_path


def create_job_context(contract_path: str):
    return JobContext(contract_path=contract_path)
