"""Contract persistence and runtime image installation service."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import yaml
from six import StringIO

from utils.config import CONTRACTS_DIR

from .contract_models import Contract


logger = logging.getLogger(__name__)


class ContractService:
    def __init__(self):
        self._pull_layers_lock = threading.Lock()
        self._pull_layers: dict[str, dict[str, dict[str, Any]]] = {}
        self._contract_operation_lock = threading.Lock()
        self._pull_state_lock = threading.Lock()
        self._pull_futures: dict[str, Any] = {}
        self._pull_executor = ThreadPoolExecutor(max_workers=1)

    def _contract_directory(self) -> Path:
        return Path(os.environ.get("MELD_CONTRACT_DIRECTORY", CONTRACTS_DIR))

    def _contract_path(self, contract_id: str) -> Path:
        return self._contract_directory() / contract_id / "contract.yaml"

    def _progress_path(self, contract_id: str) -> Path:
        return self._contract_directory() / contract_id / "status.json"

    def parse_contract(self, payload: str) -> Contract:
        if not payload.strip():
            raise ValueError("The request body must not be empty.")
        return Contract.from_yaml(StringIO(payload))

    def get_contract(self, contractId: str) -> Contract:
        with self._contract_path(contractId).open(encoding="utf-8") as contract_file:
            contract = Contract.from_yaml(contract_file)
        if contract.id != contractId:
            raise FileNotFoundError(contractId)
        return contract

    def _find_duplicate(self, contract: Contract) -> Contract | None:
        fingerprint = contract.fingerprint
        canonical_content = contract.canonical_content()
        directory = self._contract_directory()
        if not directory.exists():
            return None

        for path in sorted(directory.glob("*/contract.yaml")):
            existing_contract = Contract.from_yaml(path)
            if existing_contract.fingerprint != fingerprint:
                continue
            if existing_contract.canonical_content() == canonical_content:
                return existing_contract
        return None

    def get_contracts(self) -> list[Contract]:
        directory = self._contract_directory()
        if not directory.exists():
            return []
        return [
            self.get_contract(path.parent.name)
            for path in sorted(directory.glob("*/contract.yaml"))
        ]

    def get_contract_info(self, contractId: str) -> dict[str, Any]:
        return self._to_contract_info(self.get_contract(contractId), True)

    def get_contracts_info(self) -> list[dict[str, Any]]:
        return [
            self._to_contract_info(contract, False)
            for contract in self.get_contracts()
        ]

    def create_contract(
        self,
        contract: Contract,
        registry_api_key: str | None = None,
    ) -> bool:
        try:
            image_ref = contract.runtime.image.construct_image_ref()
        except Exception as error:
            raise ConnectionError(str(error)) from error

        with self._contract_operation_lock:
            existing_contract = self._find_duplicate(contract)
            if existing_contract is not None:
                contract.contract.id = existing_contract.id
                pull_state = self._get_pull_status(existing_contract.id)
                if not pull_state or pull_state.get("status") != "failed":
                    raise FileExistsError(existing_contract.id)
                self._start_image_pull(
                    existing_contract,
                    image_ref,
                    registry_api_key,
                )
                return False

            contract.assign_id(force=True)

            image_available = self._image_available(image_ref)
            if image_available:
                self._track_pull_progress(contract.id, "available")
            else:
                self._start_image_pull(contract, image_ref, registry_api_key)

            self._store_contract(contract)
            return image_available

    def delete_contract(self, contractId: str) -> None:
        contract = self.get_contract(contractId)
        shutil.rmtree(self._contract_path(contract.id).parent)
        from ModelManager import remove_runtime

        remove_runtime(contract)

    def _image_available(self, image_ref: str) -> bool:
        from ModelEnvironment.docker_runtime import image_exists

        try:
            return image_exists(image_ref)
        except Exception as error:
            raise ConnectionError(str(error)) from error

    def _pull_runtime_image(
        self,
        image_ref: str,
        progress_callback: Callable,
        registry_api_key: str | None = None,
    ) -> None:
        from ModelEnvironment.docker_runtime import pull_image

        pull_kwargs: dict[str, Any] = {"progress_callback": progress_callback}
        if registry_api_key:
            pull_kwargs["registry_api_key"] = registry_api_key
        pull_image(image_ref, **pull_kwargs)

    def _read_progress(self, contract_id: str) -> dict[str, Any] | None:
        path = self._progress_path(contract_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as progress_file:
                progress = json.load(progress_file)
            return progress if isinstance(progress, dict) else None
        except Exception as error:
            logger.exception(
                "Unable to read progress for contract %s: %s",
                contract_id,
                error,
            )
            return None

    def _get_pull_status(self, contract_id: str) -> dict[str, Any] | None:
        return self._read_progress(contract_id)

    def _track_pull_progress(
        self,
        contract_id: str,
        progress: dict[str, Any] | int | float | str,
        error: str | None = None,
    ) -> None:
        path = self._progress_path(contract_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existing_data = self._read_progress(contract_id)
            current_data: dict[str, Any] = {}
            if isinstance(existing_data, dict):
                for key in ("status", "progress", "error"):
                    if key in existing_data:
                        current_data[key] = existing_data[key]
            current_data.setdefault("status", "pulling")
            current_data.setdefault("progress", 0)

            if isinstance(progress, str):
                current_data["status"] = progress
                if progress == "available":
                    current_data["progress"] = 100
            elif isinstance(progress, (int, float)):
                current_data["status"] = "pulling"
                current_data["progress"] = progress
            elif isinstance(progress, dict):
                event_status = progress.get("status", "")
                layer_id = progress.get("id")
                progress_detail = progress.get("progressDetail", {})
                if layer_id and isinstance(progress_detail, dict):
                    with self._pull_layers_lock:
                        layer_info = self._pull_layers.setdefault(
                            contract_id,
                            {},
                        ).setdefault(layer_id, {})
                        if "current" in progress_detail:
                            layer_info["current"] = progress_detail["current"]
                        if "total" in progress_detail:
                            layer_info["total"] = progress_detail["total"]

                        layers = self._pull_layers[contract_id]
                        total_bytes = sum(
                            layer.get("total", 0)
                            for layer in layers.values()
                            if isinstance(layer, dict)
                        )
                        current_bytes = sum(
                            layer.get("current", 0)
                            for layer in layers.values()
                            if isinstance(layer, dict)
                        )
                    if total_bytes > 0:
                        current_data["progress"] = max(
                            current_data.get("progress", 0),
                            round((current_bytes / total_bytes) * 100, 2),
                        )
                if event_status:
                    current_data["status"] = event_status

            if progress == "available":
                with self._pull_layers_lock:
                    self._pull_layers.pop(contract_id, None)

            if error:
                current_data["status"] = "failed"
                current_data["error"] = error
            elif isinstance(progress, str) and progress in {"pulling", "available"}:
                current_data.pop("error", None)

            temp_path = path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as progress_file:
                json.dump(current_data, progress_file, indent=2)
            temp_path.replace(path)
        except Exception as error:
            logger.exception(
                "Unable to write progress for contract %s: %s",
                contract_id,
                error,
            )

    def _pull_image(
        self,
        contract_id: str,
        image_ref: str,
        registry_api_key: str | None = None,
    ) -> None:
        retry_number = 0
        while True:
            try:
                self._track_pull_progress(contract_id, "pulling")
                self._pull_runtime_image(
                    image_ref,
                    lambda event: self._track_pull_progress(contract_id, event),
                    registry_api_key,
                )
            except Exception as error:
                retry_number += 1
                logger.warning(
                    "Unable to pull image %s; retrying attempt %s: %s",
                    image_ref,
                    retry_number,
                    error,
                )
                self._track_pull_progress(contract_id, "pulling", str(error))
                time.sleep(self._pull_retry_delay(retry_number))
                continue

            self._track_pull_progress(contract_id, "available")
            return

    def _pull_retry_delay(self, retry_number: int) -> float:
        try:
            configured_delay = float(
                os.environ.get("MELD_PULL_RETRY_DELAY_SECONDS", "1")
            )
        except ValueError:
            configured_delay = 1.0
        backoff = 2 ** min(retry_number - 1, 6)
        return max(0.0, min(configured_delay * backoff, 60.0))

    def _start_image_pull(
        self,
        contract: Contract,
        image_ref: str,
        registry_api_key: str | None = None,
    ) -> None:
        self._track_pull_progress(contract.id, "pulling")
        with self._pull_state_lock:
            existing_future = self._pull_futures.get(contract.id)
            if existing_future is not None and not existing_future.done():
                return
            self._pull_futures[contract.id] = self._pull_executor.submit(
                self._pull_image,
                contract.id,
                image_ref,
                registry_api_key,
            )

    def _installation_status(self, contract: Contract) -> tuple[str, float | None]:
        pull_state = self._get_pull_status(contract.id)
        if isinstance(pull_state, dict) and pull_state.get("status") in {
            "available",
            "ready",
        }:
            return "ready", None

        if pull_state is None or pull_state.get("status") == "failed":
            try:
                if self._image_available(
                    contract.runtime.image.construct_image_ref()
                ):
                    self._track_pull_progress(contract.id, "available")
                    return "ready", None
            except Exception as error:
                logger.warning(
                    "Unable to check image availability for contract %s: %s",
                    contract.id,
                    error,
                )

        progress = pull_state.get("progress", 0) if isinstance(pull_state, dict) else 0
        if not isinstance(progress, (int, float)):
            progress = 0
        return "installing", progress

    def _to_contract_info(
        self,
        contract: Contract,
        include_contract: bool,
    ) -> dict[str, Any]:
        status, progress = self._installation_status(contract)
        if status == "installing":
            pull_state = self._get_pull_status(contract.id)
            if pull_state is None or pull_state.get("status") == "failed":
                try:
                    self._start_image_pull(
                        contract,
                        contract.runtime.image.construct_image_ref(),
                    )
                except Exception as error:
                    logger.exception(
                        "Unable to queue image installation for contract %s: %s",
                        contract.id,
                        error,
                    )

        payload: dict[str, Any] = {
            "id": contract.id,
            "status": status,
        }
        if status == "installing":
            payload["progress"] = progress
        if include_contract:
            payload["contract"] = contract.to_dict()
        return payload

    def _store_contract(self, contract: Contract) -> None:
        path = self._contract_path(contract.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as contract_file:
                yaml.safe_dump(contract.to_dict(), contract_file, sort_keys=False)
        except FileExistsError:
            raise
        except Exception:
            path.unlink(missing_ok=True)
            raise


__all__ = ["ContractService"]
