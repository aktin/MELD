from __future__ import annotations

from .job_context import ContextProvider, JobContext

__all__ = ["ContextProvider", "InferenceRunner", "JobContext", "run_inference"]


def __getattr__(name: str):
    if name in {"InferenceRunner", "run_inference"}:
        from .inference import InferenceRunner as _InferenceRunner, run_inference as _run_inference

        globals()["InferenceRunner"] = _InferenceRunner
        globals()["run_inference"] = _run_inference
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
