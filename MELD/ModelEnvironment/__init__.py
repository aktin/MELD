from __future__ import annotations

from .job_context import ContextProvider, JobContext, JobStatus

__all__ = [
    "ContextProvider",
    "JobContext",
    "JobStatus",
    "InferenceRunner",
    "create_container",
    "delete_image",
    "ensure_image_exists",
    "get_image_size",
    "pull_image",
    "run_inference",
    "start_container",
    "stop_container",
    "wait_for_container",
]

_INFERENCE_EXPORTS = {"InferenceRunner", "run_inference"}
_RUNTIME_EXPORTS = {
    "create_container",
    "delete_image",
    "ensure_image_exists",
    "get_image_size",
    "pull_image",
    "start_container",
    "stop_container",
    "wait_for_container",
}


def __getattr__(name: str):
    if name in _INFERENCE_EXPORTS:
        from .inference import (
            InferenceRunner as _InferenceRunner,
            run_inference as _run_inference,
        )

        globals()["InferenceRunner"] = _InferenceRunner
        globals()["run_inference"] = _run_inference
        return globals()[name]

    if name in _RUNTIME_EXPORTS:
        from . import docker_runtime

        value = getattr(docker_runtime, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
