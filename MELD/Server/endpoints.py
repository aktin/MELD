"""Assemble and expose the MELD API routes.

Route implementations live in modules grouped by API responsibility. This
module remains the public import location for the application blueprint and
the existing API symbols.
"""

from .api import (
    API_VERSION,
    MeldApi,
    api,
    bp,
    contract_body,
    contracts,
    error_model,
    error_response,
    inference_body,
    inferences,
    not_implemented,
    schedule_body,
    schedules,
)
from .contracts import (
    ContractResource,
    ContractValidationResource,
    ContractsResource,
)
from .inferences import (
    InferenceLogsResource,
    InferenceResource,
    InferencesResource,
)
from .schedules import ContractScheduleResource, ContractSchedulesResource
from .system import health, version

__all__ = [
    "API_VERSION",
    "ContractResource",
    "ContractValidationResource",
    "ContractsResource",
    "ContractScheduleResource",
    "ContractSchedulesResource",
    "InferenceLogsResource",
    "InferenceResource",
    "InferencesResource",
    "MeldApi",
    "api",
    "bp",
    "contract_body",
    "contracts",
    "error_model",
    "error_response",
    "health",
    "inference_body",
    "inferences",
    "not_implemented",
    "schedule_body",
    "schedules",
    "version",
]
