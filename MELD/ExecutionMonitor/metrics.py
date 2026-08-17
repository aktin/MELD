from enum import StrEnum

# TODO KSI?
class Metrics(StrEnum):
    TOTAL_EXECUTION_TIME = "total_execution_time"
    TOTAL_EXECUTION_START_TIMESTAMP = "total_execution_start_timestamp"
    TOTAL_EXECUTION_END_TIMESTAMP = "total_execution_end_timestamp"

    QUERY_EXECUTION_TIME = "query_execution_time"
    QUERY_EXECUTION_START_TIMESTAMP = "query_execution_start_timestamp"
    QUERY_EXECUTION_END_TIMESTAMP = "query_execution_end_timestamp"

    INFERENCE_TIME = "inference_time"
    INFERENCE_START_TIMESTAMP = "inference_start_timestamp"
    INFERENCE_END_TIMESTAMP = "inference_end_timestamp"

    INPUT_COPY_TIME = "input_copy_time"
    INPUT_COPY_START_TIMESTAMP = "input_copy_start_timestamp"
    INPUT_COPY_END_TIMESTAMP = "input_copy_end_timestamp"

    OUTPUT_COPY_TIME = "output_copy_time"
    OUTPUT_COPY_START_TIMESTAMP = "output_copy_start_timestamp"
    OUTPUT_COPY_END_TIMESTAMP = "output_copy_end_timestamp"

    ARCHIVE_PACKING_TIME = "archive_packing_time"
    ARCHIVE_PACKING_START_TIMESTAMP = "archive_packing_start_timestamp"
    ARCHIVE_PACKING_END_TIMESTAMP = "archive_packing_end_timestamp"

    JOB_ID = "job_id"
    CONTRACT_ID = "contract_id"
    CONTRACT_VERSION = "contract_version"
    ORCHESTRATOR_VERSION = "orchestrator_version"

    DOCKER_IMAGE_DIGEST = "docker_image_digest"
    DOCKER_IMAGE_NAME = "docker_image_name"
    DOCKER_IMAGE_TAG = "docker_image_tag"
    DOCKER_IMAGE_SIZE = "docker_image_size"

    EXPECTED_FEATURE_COUNT = "expected_feature_count"
    EXPECTED_PREDICTOR_COUNT = "expected_predictor_count"
    ACTUAL_FEATURE_COUNT = "actual_feature_count"
    ACTUAL_PREDICTOR_COUNT = "actual_predictor_count"

    RUNTIME_EXIT_CODE = "runtime_exit_code"
    QUERY_RESULT_ROW_COUNT = "query_result_row_count"
    INFERENCE_RESULT_ROW_COUNT = "inference_result_row_count"
    INFERENCE_INPUT_ROW_COUNT = "inference_input_row_count"
    INFERENCE_RESULT_SIZE = "inference_result_size"
    INFERENCE_INPUT_SIZE = "inference_input_size"
