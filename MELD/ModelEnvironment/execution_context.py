import datetime
import json
import threading
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from Logger import get_job_logger
from ModelManager.contract import Contract, construct_image_ref
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
        input_data_path (Path): Path to the input data folder for the job.
        output_data_path (Path): Path to the output data folder for the job.
        status_path (Path): Path to the status folder for the job.
        logs_path (Path): Path to the logs folder for the job.
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
        return construct_image_ref(self.contract.runtime.image)

    def _create_input_folder(self) -> Path:
        input_path = self._job_folder / "input"
        input_path.mkdir(parents=True, exist_ok=True)
        return input_path

    def _create_output_folder(self) -> Path:
        output_path = self._job_folder / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _create_job_folder(self) -> Path:
        job_folder = self._root_path / CONTRACTS_DIR / self.contract.id / "executions" / self.execution_id
        job_folder.mkdir(parents=True, exist_ok=True)
        return job_folder

    def _create_execution_id(self):
        return f"{self.contract.id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    def set_status(self, status: ExecutionStatus):
        self.status = status
        with (self.status_path / "status.json").open("w") as f:
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

    def _create_status_folder(self) -> Path:
        status_path = self._job_folder / "status"
        status_path.mkdir(parents=True, exist_ok=True)
        return status_path

    def _create_log_folder(self) -> Path:
        log_path = self._job_folder / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    @property
    def _root_path(self):
        """
        Retrieves the root path for the application. If the `MELD_ROOT_DIR` environment
        variable is not set, it assumes it is running in a Docker container and uses the default value "/".

        Returns:
            Path: The root path, either from the `MELD_ROOT_DIR` environment
            variable or the default value "/".
        """
        return Path(ROOT_DIR)

    @staticmethod
    def create(contract: str | Contract, execution_id: str = None) -> ExecutionContext:
        if isinstance(contract, str):
            contract = read_contract(contract_id=contract)
        return ExecutionContext(contract=contract, execution_id=execution_id)

    def _read_status(self):
        with (self.status_path / "status.json").open("r") as f:
            self.status = json.loads(f.read())


class ContextProvider:
    def __init__(self, job_context: ExecutionContext):
        self.job_context = job_context

    def get(self) -> ExecutionContext:
        return self.job_context
