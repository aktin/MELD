"""Inference endpoints."""

from flask_restx import Resource

from .api import (
    error_model,
    inference_body,
    inferences as inferences_namespace,
    not_implemented,
)


@inferences_namespace.route("", strict_slashes=False)
@inferences_namespace.doc(params={"contractId": "Contract identifier"})
class InferencesResource(Resource):
    @inferences_namespace.doc(description="List inferences for a contract.")
    @inferences_namespace.response(200, "List of inferences")
    @inferences_namespace.response(404, "Contract does not exist", error_model)
    def get(self, contractId):
        return not_implemented()

    @inferences_namespace.doc(description="Start an inference asynchronously.")
    @inferences_namespace.expect(inference_body, validate=False)
    @inferences_namespace.response(202, "Inference started")
    @inferences_namespace.response(400, "Inference request is malformed", error_model)
    @inferences_namespace.response(404, "Contract does not exist", error_model)
    def post(self, contractId):
        return not_implemented()


@inferences_namespace.route("/<string:inferenceId>")
@inferences_namespace.doc(
    params={
        "contractId": "Contract identifier",
        "inferenceId": "Inference identifier",
    }
)
class InferenceResource(Resource):
    @inferences_namespace.doc(
        description=(
            "Retrieve inference status while running or the completed result "
            "archive as a file object."
        )
    )
    @inferences_namespace.response(200, "Inference status or completed result archive")
    @inferences_namespace.response(404, "Contract or inference does not exist", error_model)
    def get(self, contractId, inferenceId):
        return not_implemented()

    @inferences_namespace.doc(description="Cancel a running inference.")
    @inferences_namespace.response(204, "Inference canceled")
    @inferences_namespace.response(404, "Contract or inference does not exist", error_model)
    def delete(self, contractId, inferenceId):
        return not_implemented()


@inferences_namespace.route("/<string:inferenceId>/logs")
@inferences_namespace.doc(
    params={
        "contractId": "Contract identifier",
        "inferenceId": "Inference identifier",
    }
)
class InferenceLogsResource(Resource):
    @inferences_namespace.doc(description="Retrieve the inference log stream.")
    @inferences_namespace.response(200, "Inference log stream")
    @inferences_namespace.response(404, "Contract or inference does not exist", error_model)
    def get(self, contractId, inferenceId):
        return not_implemented()
