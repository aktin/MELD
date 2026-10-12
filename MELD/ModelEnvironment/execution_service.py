import json
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Generator, Protocol

from ModelEnvironment import ExecutionContext, ExecutionStatus
from ModelManager import run_inference
from ModelManager.contract import Contract
from utils.config import CONTRACTS_DIR

FINAL_EXECUTION_STATES = {
    ExecutionStatus.SUCCESS.value,
    ExecutionStatus.FAILED.value,
    ExecutionStatus.CANCELED.value,
    ExecutionStatus.TIMEOUT.value,
}


class ExecutionsResult(Protocol):
    execution_id: str
    status: ExecutionStatus


class ExecutionService:
    """Run at most one inference at a time and track its lifecycle."""

    def __init__(self):
        self.current_inference: Future | None = None
        self.current_execution_id: str | None = None
        self.current_context: ExecutionContext | None = None
        self._execution_tasks: dict[str, tuple[Future, ExecutionContext]] = {}
        self._execution_lock = threading.RLock()
        self._execution_executor = ThreadPoolExecutor(max_workers=1)

    def get_executions(self, contract_id: str) -> list[ExecutionsResult]:
        self._check_contract_exists(contract_id)

        statuses = list(self._get_contract_folder(contract_id).glob("executions/**/status.json"))

        if len(statuses) == 0:
            return []

        executions = []

        for s in statuses:
            status = json.loads(s.read_text(encoding="utf-8"))
            execution_id = s.parent.parent.name

            executions.append({"status": status, "execution_id": execution_id})
        return executions

    def get_execution(self, contract_id: str, execution_id: str) -> ExecutionContext | None:
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        ctx = ExecutionContext.create(contract_id, execution_id)
        return ctx

    def start_execution(self, contract_id: str) -> str:
        """Create a persistent context and submit its inference in the background."""
        self._check_contract_exists(contract_id)
        contract = Contract.from_yaml(self._get_contract_path(contract_id))
        ctx = ExecutionContext.create(contract_id)
        with self._execution_lock:
            # Store the future and context before returning so cancellation can
            # address work that has been queued or has already started running.
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
        """Remove a completed task without clearing state for a newer task."""
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

    def cancel_execution(self, contract_id: str, execution_id: str, wait: bool = False) -> bool:
        """Request cooperative cancellation for a queued, active, or persisted execution.

        Returns ``False`` when the execution has already completed. A queued
        future can be canceled immediately. A running inference must
        observe the context's cancellation event and finish its cleanup first.
        Waiting is used by contract deletion so final status and archive writes
        cannot race with removal of the execution directory.
        """
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        future_to_wait = None
        with self._execution_lock:
            # Normally tasks are found in the map. The current fields are kept
            # as a fallback for the brief period around done-callback cleanup.
            task = self._execution_tasks.get(execution_id)
            if task is None and self.current_execution_id == execution_id:
                if self.current_context is not None and self.current_inference is not None:
                    task = (self.current_inference, self.current_context)

            if task is not None:
                future, ctx = task
                status = _status_value(ctx)
                if status in FINAL_EXECUTION_STATES or future.done() and not future.cancelled():
                    return False

                ctx.request_cancel()
                ctx.set_status(ExecutionStatus.CANCELED)
                if future.cancel():
                    # The task had not started, so no inference cleanup needs
                    # to run before the cancellation request is complete.
                    self._execution_tasks.pop(execution_id, None)
                    if self.current_inference is future:
                        self.current_inference = None
                        self.current_execution_id = None
                        self.current_context = None
                else:
                    # A running future cannot be force-canceled. Its context
                    # callback stops runtime work and run_inference performs
                    # the remaining cleanup in its finally block.
                    future_to_wait = future

        if future_to_wait is not None and wait:
            # Contract deletion uses this barrier before deleting its storage.
            future_to_wait.result()
            return

        if task is not None:
            return True

        # The process may have restarted since this execution was created, so
        # reconstruct its context from the persisted execution directory.
        ctx = ExecutionContext.create(contract_id, execution_id)
        if _status_value(ctx) in FINAL_EXECUTION_STATES:
            return False

        ctx.request_cancel()
        ctx.set_status(ExecutionStatus.CANCELED)
        return True

    def get_result_archive(self, contract_id: str, execution_id: str):
        """Read a completed execution archive into an independent byte stream."""
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        execution_folder = self._get_execution_folder(contract_id, execution_id)
        archive_path = execution_folder / "output" / "summarized_execution.zip"

        try:
            archive_file = archive_path.open("rb")
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
        return self._get_execution_folder(contract_id, execution_id).exists()

    def _get_execution_folder(self, contract_id: str, execution_id: str) -> Path:
        return self._get_contract_folder(contract_id) / "executions" / execution_id

    def _check_contract_exists(self, contract_id: str) -> None:
        if not self._contract_exists(contract_id):
            raise FileNotFoundError(f"Contract {contract_id} does not exist")

    def _contract_exists(self, contract_id: str) -> bool:
        return self._get_contract_path(contract_id).exists()

    def _get_contract_path(self, contract_id: str) -> Path:
        return self._get_contract_folder(contract_id) / "contract.yaml"

    def _get_contract_folder(self, contract_id: str) -> Path:
        return Path(CONTRACTS_DIR) / contract_id

    def read_log(self, contract_id: str, execution_id: str) -> str:
        """Read the first persisted log file for an execution."""
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        ctx = ExecutionContext.create(contract_id, execution_id)

        log_files = list(Path(ctx.logs_path).glob("*.log"))
        if not log_files:
            raise FileNotFoundError(f"No log file found for execution {execution_id}")

        with log_files[0].open("r", encoding="utf-8") as f:
            return f.read()

    def stream_log(self, contract_id: str, execution_id: str) -> Generator[str, None, None]:
        """Yield existing log lines and continue waiting for appended lines."""
        self._check_contract_exists(contract_id)
        self._check_execution_exists(contract_id, execution_id)

        ctx = ExecutionContext.create(contract_id, execution_id)

        log_files = list(Path(ctx.logs_path).glob("*.log"))
        if not log_files:
            raise FileNotFoundError(f"No log file found for execution {execution_id}")

        log_file = log_files[0]

        def read_lines() -> Generator[str, None, None]:
            with log_file.open("r", encoding="utf-8") as f:
                while True:
                    line = f.readline()

                    if line:
                        yield line.rstrip("\n")
                    else:
                        # Inference writes logs while this generator is consumed.
                        time.sleep(0.1)

        return read_lines()


def _status_value(ctx) -> str | None:
    """Normalize persisted and in-memory context status representations."""
    status = ctx.status
    if isinstance(status, dict):
        return status.get("status")
    if isinstance(status, ExecutionStatus):
        return status.value
    return status
