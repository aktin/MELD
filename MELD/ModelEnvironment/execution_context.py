import datetime
import json
import os
import threading
from collections.abc import Callable
from enum import Enum

from Logger import get_job_logger
from ModelManager.contract_models import Contract
from utils.config import ROOT_DIR, CONTRACTS_DIR
from utils.utils import read_contract


class ExecutionStatus(Enum):
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


class ExecutionContext:
    """
    Represents a job context within the application.

    The JobContext class is responsible for managing the lifecycle and setup of a
    job, including folder structure, logging, and status updates. It initializes
    based on a contract, which defines runtime configurations, input/output schemas,
    and other job-specific parameters.

    Attributes:
        status (ExecutionStatus): The current status of the job.
        contract (Contract): The loaded job contract, including runtime and input schema configurations.
        execution_id (str): A unique identifier for the job, generated based on the current timestamp.
        input_data_path (str): Path to the input data folder for the job.
        output_data_path (str): Path to the output data folder for the job.
        status_path (str): Path to the status folder for the job.
        logs_path (str): Path to the logs folder for the job.
        logger (logging.Logger): The logger used for job-related logging.
    """

    def __init__(self, contract: Contract, execution_id: str = None):
        self.status = None
        self._cancel_event = threading.Event()
        self._cancel_callback: Callable[[], None] | None = None
        self._cancel_lock = threading.Lock()

        self.contract = contract

        self.execution_id = execution_id if execution_id is not None else self._create_execution_id()

        # set up folder structure
        self._job_folder = self._create_job_folder()
        self.input_data_path = self._create_input_folder()
        self.output_data_path = self._create_output_folder()
        self.status_path = self._create_status_folder()
        self.logs_path = self._create_log_folder()

        self.logger = get_job_logger(self.execution_id, self.logs_path)

        if execution_id is not None:
            self._read_status()
        else:
            self.logger.info(f"Job {self.execution_id} created")
            self.set_status(ExecutionStatus.PENDING)


    @property
    def image_ref(self):
        return self.contract.runtime.image.construct_image_ref()

    def _create_input_folder(self):
        input_path = os.path.join(self._job_folder, "input")

        if not os.path.exists(input_path):
            os.makedirs(input_path)

        return input_path

    def _create_output_folder(self):
        output_path = os.path.join(self._job_folder, "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path

    def _create_job_folder(self):
        job_folder = os.path.join(self._root_path, CONTRACTS_DIR, self.contract.id, "executions", self.execution_id)

        if not os.path.exists(job_folder):
            os.makedirs(job_folder)

        return job_folder

    def _create_execution_id(self):
        return f"{self.contract.id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def set_status(self, status: ExecutionStatus):
        self.status = status
        with open(os.path.join(self.status_path, "status.json"), "w") as f:
            json.dump({
                "status": status.value,
                "lastUpdated": datetime.datetime.now().isoformat(),
            }, f, sort_keys=True, indent=4)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def request_cancel(self) -> None:
        with self._cancel_lock:
            self._cancel_event.set()
            callback = self._cancel_callback

        if callback is not None:
            callback()

    def set_cancel_callback(self, callback: Callable[[], None] | None) -> None:
        with self._cancel_lock:
            self._cancel_callback = callback
            cancel_requested = self._cancel_event.is_set()

        if callback is not None and cancel_requested:
            callback()

    def _create_status_folder(self):
        status_path = os.path.join(self._job_folder, "status")

        if not os.path.exists(status_path):
            os.makedirs(status_path)

        return status_path

    def _create_log_folder(self):
        log_path = os.path.join(self._job_folder, "logs")

        if not os.path.exists(log_path):
            os.makedirs(log_path)

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
    def create(contract_id: str, execution_id: str = None) -> ExecutionContext:
        contract = read_contract(contract_id=contract_id)
        return ExecutionContext(contract=contract, execution_id=execution_id)

    def _read_status(self):
        with open(os.path.join(self.status_path, "status.json"), "r") as f:
            self.status = json.loads(f.read())


class ContextProvider:
    def __init__(self, job_context: ExecutionContext):
        self.job_context = job_context

    def get(self) -> ExecutionContext:
        return self.job_context
