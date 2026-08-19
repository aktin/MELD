import datetime
import os.path

from Logger import get_job_logger
from ModelManager import config_loader
from utils import construct_image_ref


class JobContext:
    """
    Represents a job context within the application.

    The JobContext class is responsible for managing the lifecycle and setup of a
    job, including folder structure, logging, and status updates. It initializes
    based on a contract, which defines runtime configurations, input/output schemas,
    and other job-specific parameters.

    Attributes:
        container_status (str): The current status of the job container, if applicable.
        contract_path (str): The file path to the job's contract.
        contract (dict): The loaded job contract, including runtime and input schema configurations.
        job_id (str): A unique identifier for the job, generated based on the current timestamp.
        input_data_path (str): Path to the input data folder for the job.
        output_data_path (str): Path to the output data folder for the job.
        logs_path (str): Path to the logs folder for the job.
        logger (logging.Logger): The logger used for job-related logging.
    """

    def __init__(self, contract_path: str):
        self.container_status = None

        self.contract_path = contract_path
        self.contract = config_loader.load_contract(contract_path)

        self.job_id = self._create_job_id()

        # set up folder structure
        self._job_folder = self._create_job_folder()
        self.input_data_path = self._create_input_folder()
        self.output_data_path = self._create_output_folder()
        self.reports_path = self._create_reports_folder()
        self.logs_path = self._create_log_folder()

        self.logger = get_job_logger(self.job_id, self.logs_path)

    @property
    def contract_path(self):
        return self._contract_path

    @contract_path.setter
    def contract_path(self, value):
        self._contract_path = value
        if not os.path.exists(value):
            raise FileNotFoundError(f"Contract file {value} does not exist")

    @property
    def image_ref(self):
        return construct_image_ref(self.contract)

    def _create_input_folder(self):
        return self._create_folder(self._job_folder, "input")

    def _create_output_folder(self):
        return self._create_folder(self._job_folder, "output")

    def _create_job_folder(self):
        job_folder = os.path.join(self._root_path, "jobs")
        return self._create_folder(job_folder, self.job_id)

    def _create_reports_folder(self):
        return self._create_folder(self._job_folder, "reports")

    def _create_folder(self, base_path: str, folder_name: str):
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            raise FileExistsError(f"Folder {folder_path} already exists")

        return folder_path

    def _create_job_id(self):
        return f"{self.contract['contract']['id']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _create_log_folder(self):
        return self._create_folder(self._job_folder, "logs")

    @property
    def _root_path(self):
        """
        Retrieves the root path for the application. If the `MELD_ROOT_DIR` environment
        variable is not set, it assumes it is running in a Docker container and uses the default value "/".

        Returns:
            str: The root path, either from the `MELD_ROOT_DIR` environment
            variable or the default value "/".
        """
        return os.environ.get("MELD_ROOT_DIR", "/")

    @staticmethod
    def create_job_context(contract_path: str):
        context = JobContext(contract_path=contract_path)
        return context

class ContextProvider:
    def __init__(self, job_context: JobContext):
        self.job_context = job_context

    def get(self) -> JobContext:
        return self.job_context
