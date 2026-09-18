import json
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
import glob
import os
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

from ModelEnvironment import ExecutionContext, ExecutionStatus
from ModelManager import Contract, run_inference
from utils.config import CONTRACTS_DIR
from werkzeug.exceptions import BadRequest


FINAL_EXECUTION_STATES = {
    ExecutionStatus.SUCCESS.value,
    ExecutionStatus.FAILED.value,
    ExecutionStatus.CANCELED.value,
    ExecutionStatus.TIMEOUT.value,
}


class ExecutionService:
    def __init__(self):
        self.current_inference: Future | None = None
        self.current_execution_id: str | None = None
        self.current_context: ExecutionContext | None = None
        self._execution_tasks: dict[str, tuple[Future, ExecutionContext]] = {}
        self._execution_lock = threading.RLock()
        self._execution_executor = ThreadPoolExecutor(max_workers=1)

    def get_executions(self, contract_id: str) -> Generator[dict[str, str], Any, list[Any] | None]:
        self._check_contract_exists(contract_id)

        statuses = glob.glob(os.path.join(self._get_contract_folder(contract_id), "executions", "**", "status.json"), recursive=True)

        if len(statuses) == 0:
            return []

        for s in sorted(statuses, key=lambda status_path: Path(status_path).parent.parent.name, reverse=True):
            path = Path(s)
            status = json.loads(path.read_text(encoding="utf-8"))
            execution_id = path.parent.parent.name

            yield {"status": status, "execution_id": execution_id}

    def get_execution(self, contract_id: str, execution_id: str) -> ExecutionContext | None:
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        ctx = ExecutionContext.create(contract_id, execution_id)
        return ctx

    def start_execution(self, contract_id: str) -> str:
        self._check_contract_exists(contract_id)
        contract = Contract.from_yaml(self._get_contract_path(contract_id))
        ctx = ExecutionContext.create(contract_id)
        with self._execution_lock:
            future = self._execution_executor.submit(run_inference, contract, ctx)
            self._execution_tasks[ctx.execution_id] = (future, ctx)
            self.current_inference = future
            self.current_execution_id = ctx.execution_id
            self.current_context = ctx
            future.add_done_callback(self._clear_current_inference)

        return ctx.execution_id

    async def create_execution(self, contract_id: str) -> str:
        return self.start_execution(contract_id)

    def _clear_current_inference(self, future: Future) -> None:
        with self._execution_lock:
            execution_id = next(
                (
                    execution_id
                    for execution_id, (tracked_future, _) in self._execution_tasks.items()
                    if tracked_future is future
                ),
                None,
            )
            if execution_id is not None:
                del self._execution_tasks[execution_id]

            if self.current_inference is future:
                self.current_inference = None
                self.current_execution_id = None
                self.current_context = None

    def cancel_execution(self, contract_id: str, execution_id: str):
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        with self._execution_lock:
            task = self._execution_tasks.get(execution_id)
            if task is None and self.current_execution_id == execution_id:
                if self.current_context is not None and self.current_inference is not None:
                    task = (self.current_inference, self.current_context)

            if task is not None:
                future, ctx = task
                status = _status_value(ctx)
                if status in FINAL_EXECUTION_STATES or future.done() and not future.cancelled():
                    raise BadRequest(
                        f"Execution {execution_id} has already completed."
                    )

                ctx.request_cancel()
                ctx.set_status(ExecutionStatus.CANCELED)
                if future.cancel():
                    self._execution_tasks.pop(execution_id, None)
                    if self.current_inference is future:
                        self.current_inference = None
                        self.current_execution_id = None
                        self.current_context = None
                return

        ctx = ExecutionContext.create(contract_id, execution_id)
        if _status_value(ctx) in FINAL_EXECUTION_STATES:
            raise BadRequest(f"Execution {execution_id} has already completed.")

        ctx.request_cancel()
        ctx.set_status(ExecutionStatus.CANCELED)


    def get_result_archive(self, contract_id: str, execution_id: str):
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        execution_folder = self._get_execution_folder(contract_id, execution_id)
        archive_path = os.path.join(execution_folder, "output", "summarized_execution.zip")

        try:
            archive_file = open(archive_path, "rb")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Result archive for execution {execution_id} does not exist"
            ) from error

        with archive_file as zip_ref:
            return BytesIO(zip_ref.read())

    def _check_execution_exists(self, contract_id: str, execution_id: str) -> None:
        if not self._execution_exists(contract_id, execution_id):
            raise FileNotFoundError(f"Execution {execution_id} does not exist")

    def _execution_exists(self, contract_id: str, execution_id: str) -> bool:
        return os.path.exists(self._get_execution_folder(contract_id, execution_id))

    def _get_execution_folder(self, contract_id: str, execution_id: str) -> str:
        return os.path.join(self._get_contract_folder(contract_id), "executions", execution_id)

    def _check_contract_exists(self, contract_id: str) -> None:
        if not self._contract_exists(contract_id):
            raise FileNotFoundError(f"Contract {contract_id} does not exist")

    def _contract_exists(self, contract_id: str) -> bool:
        return os.path.exists(self._get_contract_path(contract_id))

    def _get_contract_path(self, contract_id: str) -> str:
        return os.path.join(self._get_contract_folder(contract_id), "contract.yaml")

    def _get_contract_folder(self, contract_id: str) -> str:
        return os.path.join(CONTRACTS_DIR, contract_id)

    def read_log(self, contract_id: str, execution_id: str) -> str:
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        ctx = ExecutionContext.create(contract_id, execution_id)

        log_files = glob.glob(os.path.join(ctx.logs_path, "*.log"))
        if not log_files:
            raise FileNotFoundError(f"No log file found for execution {execution_id}")

        with open(log_files[0], "r", encoding="utf-8") as f:
            return f.read()

    def stream_log(self, contract_id: str, execution_id: str) -> Generator[str, None, None]:
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        ctx = ExecutionContext.create(contract_id, execution_id)

        log_files = glob.glob(os.path.join(ctx.logs_path, "*.log"))
        if not log_files:
            raise FileNotFoundError(f"No log file found for execution {execution_id}")

        log_file = log_files[0]

        def read_lines() -> Generator[str, None, None]:
            with open(log_file, "r", encoding="utf-8") as f:
                while True:
                    line = f.readline()

                    if line:
                        yield line.rstrip("\n")
                    else:
                        time.sleep(0.1)

        return read_lines()


def _status_value(ctx) -> str | None:
    status = ctx.status
    if isinstance(status, dict):
        return status.get("status")
    if isinstance(status, ExecutionStatus):
        return status.value
    return status
