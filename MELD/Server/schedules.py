"""Schedule endpoints."""

from flask_restx import Resource

from .api import (
    error_model,
    not_implemented,
    schedule_body,
    schedules as schedules_namespace,
)


@schedules_namespace.route("", strict_slashes=False)
@schedules_namespace.doc(params={"contractId": "Contract identifier"})
class ContractSchedulesResource(Resource):
    @schedules_namespace.doc(description="List schedules for a contract.")
    @schedules_namespace.response(200, "List of schedules")
    @schedules_namespace.response(404, "Contract does not exist", error_model)
    def get(self, contractId):
        return not_implemented()

    @schedules_namespace.doc(description="Create a schedule for a contract.")
    @schedules_namespace.expect(schedule_body, validate=False)
    @schedules_namespace.response(201, "Schedule created")
    @schedules_namespace.response(400, "Schedule is malformed or invalid", error_model)
    @schedules_namespace.response(404, "Contract does not exist", error_model)
    def post(self, contractId):
        return not_implemented()


@schedules_namespace.route("/<string:scheduleId>")
@schedules_namespace.doc(
    params={
        "contractId": "Contract identifier",
        "scheduleId": "Schedule identifier",
    }
)
class ContractScheduleResource(Resource):
    @schedules_namespace.doc(description="Retrieve a schedule.")
    @schedules_namespace.response(200, "Schedule information")
    @schedules_namespace.response(404, "Contract or schedule does not exist", error_model)
    def get(self, contractId, scheduleId):
        return not_implemented()

    @schedules_namespace.doc(description="Delete a schedule.")
    @schedules_namespace.response(204, "Schedule deleted")
    @schedules_namespace.response(404, "Contract or schedule does not exist", error_model)
    def delete(self, contractId, scheduleId):
        return not_implemented()
