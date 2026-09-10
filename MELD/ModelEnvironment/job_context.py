import datetime
import json
import os
from enum import Enum

from Logger import get_job_logger
from ModelManager import Contract
from utils.config import ROOT_DIR


class JobStatus(Enum):
    DELETING_IMAGE = "DELETING_IMAGE"
    IMAGE_DELETED = "IMAGE_DELETED"
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
    """
    Represents a job context within the application.

    The JobContext class is responsible for managing the lifecycle and setup of a
    job, including folder structure, logging, and status updates. It initializes
    based on a contract, which defines runtime configurations, input/output schemas,
    and other job-specific parameters.

    Attributes:
        status (JobStatus): The current status of the job.
        container_status (str): The current status of the job container, if applicable.
        contract (Contract): The loaded job contract, including runtime and input schema configurations.
        job_id (str): A unique identifier for the job, generated based on the current timestamp.
        input_data_path (str): Path to the input data folder for the job.
        output_data_path (str): Path to the output data folder for the job.
        status_path (str): Path to the status folder for the job.
        logs_path (str): Path to the logs folder for the job.
        logger (logging.Logger): The logger used for job-related logging.
    """

    def __init__(self, contract: Contract):
        self.status = None
        self.container_status = None

        self.contract = contract

        self.job_id = self._create_job_id()

        # set up folder structure
        self._job_folder = self._create_job_folder()
        self.input_data_path = self._create_input_folder()
        self.output_data_path = self._create_output_folder()
        self.status_path = self._create_status_folder()
        self.logs_path = self._create_log_folder()

        self.logger = get_job_logger(self.job_id, self.logs_path)
        self.logger.info(f"Job {self.job_id} created")
        self.set_status(JobStatus.PENDING)

    @property
    def image_ref(self):
        return self.contract.runtime.image.construct_image_ref()

    def _create_input_folder(self):
        input_path = os.path.join(self._job_folder, "input")

        if not os.path.exists(input_path):
            os.makedirs(input_path)
        else:
            raise FileExistsError(f"Input folder for job {self.job_id} already exists")

        return input_path

    def _create_output_folder(self):
        output_path = os.path.join(self._job_folder, "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            raise FileExistsError(f"Output folder for job {self.job_id} already exists")

        return output_path

    def _create_job_folder(self):
        job_folder = os.path.join(self._root_path, "jobs", self.job_id)

        if not os.path.exists(job_folder):
            os.makedirs(job_folder)
        else:
            raise FileExistsError(f"Job folder for job {self.job_id} already exists")

        return job_folder

    def _create_job_id(self):
        return f"{self.contract.id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    def set_status(self, status: JobStatus):
        pass
        # self.status = status
        # with open(os.path.join(self.status_path, "status.json"), "w") as f:
        #     json.dump({
        #         "status": status.value,
        #         "lastUpdated": datetime.datetime.now().isoformat(),
        #         "jobId": self.job_id
        #     }, f, sort_keys=True, indent=4)

    def _create_status_folder(self):
        status_path = os.path.join(self._job_folder, "status")

        if not os.path.exists(status_path):
            os.makedirs(status_path)
        else:
            raise FileExistsError(f"Status folder for job {self.job_id} already exists")

        return status_path

    def _create_log_folder(self):
        log_path = os.path.join(self._job_folder, "logs")

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            raise FileExistsError(f"Log folder for job {self.job_id} already exists")

        return log_path

    @property
    def _root_path(self):
        """
        Retrieves the root path for the application. If the `MELD_ROOT_DIR` environment
        variable is not set, it assumes it is running in a Docker container and uses the default value "/".

        Returns:
            str: The root path, either from the `MELD_ROOT_DIR` environment
            variable or the default value "/".
        """
        return ROOT_DIR

    @staticmethod
    def create_job_context(contract: Contract):
        context = JobContext(contract=contract)
        return context

class ContextProvider:
    def __init__(self, job_context: JobContext):
        self.job_context = job_context

    def get(self) -> JobContext:
        return self.job_context
