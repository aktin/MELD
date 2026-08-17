import datetime

import __version__
from ExecutionMonitor.metrics import Metrics
from ModelEnvironment.job_context import ContextProvider

class Timer:
    start_metric: Metrics
    start_timestamp: datetime.datetime
    end_metric: Metrics
    end_timestamp: datetime.datetime
    total_execution_time: datetime.timedelta

    def __init__(self, start_metric: Metrics, end_metric: Metrics):
        self.start_metric = start_metric
        self.end_metric = end_metric
        self.start_timestamp = datetime.datetime.now()
        self.end_timestamp = None
        self.total_execution_time = None


    def stop_timer(self) -> datetime.timedelta:
        self.end_timestamp = datetime.datetime.now()
        self.total_execution_time = self.end_timestamp - self.start_timestamp
        return self.total_execution_time

class ExecutionMonitor:
    def __init__(self, provider: ContextProvider):
        self.metrics: dict[Metrics, float | int | str] = {}
        self.timers: dict[Metrics, Timer] = {}
        self.update_metric_value(Metrics.JOB_ID, provider.get().job_id)
        self.update_metric_value(Metrics.CONTRACT_ID, provider.get().contract["contract"]["id"])
        self.update_metric_value(Metrics.CONTRACT_VERSION, provider.get().contract["contract"]["version"])
        self.update_metric_value(Metrics.EXPECTED_FEATURE_COUNT, len(provider.get().contract["input_schema"]["features"]))
        self.update_metric_value(Metrics.EXPECTED_PREDICTOR_COUNT, len(provider.get().contract["output_schema"]["predictor"]))
        self.update_metric_value(Metrics.DOCKER_IMAGE_TAG, provider.get().contract["runtime"]["image"]["tag"])
        self.update_metric_value(Metrics.DOCKER_IMAGE_NAME, provider.get().contract["runtime"]["image"]["name"])
        self.update_metric_value(Metrics.DOCKER_IMAGE_DIGEST, provider.get().contract["runtime"]["image"]["digest"])
        self.update_metric_value(Metrics.ORCHESTRATOR_VERSION, __version__.version)

    def update_metric_value(self, metric: Metrics, value: float | int | str):
        self.metrics[metric] = value

    def get_metric_value(self, metric: Metrics):
        return self.metrics[metric]

    def collect_metrics(self) -> dict[Metrics, float | int | str]:
        return self.metrics

    def start_total_execution_time(self):
        self._start_timer(Metrics.TOTAL_EXECUTION_TIME, Metrics.TOTAL_EXECUTION_START_TIMESTAMP, Metrics.TOTAL_EXECUTION_END_TIMESTAMP)

    def stop_total_execution_time(self):
        return self._stop_timer(Metrics.TOTAL_EXECUTION_TIME)

    def start_query_execution_time(self):
        self._start_timer(Metrics.QUERY_EXECUTION_TIME, Metrics.QUERY_EXECUTION_START_TIMESTAMP, Metrics.QUERY_EXECUTION_END_TIMESTAMP)

    def stop_query_execution_time(self):
        return self._stop_timer(Metrics.QUERY_EXECUTION_TIME)

    def start_inference_time(self):
        self._start_timer(Metrics.INFERENCE_TIME, Metrics.INFERENCE_START_TIMESTAMP, Metrics.INFERENCE_END_TIMESTAMP)

    def stop_inference_time(self):
        return self._stop_timer(Metrics.INFERENCE_TIME)

    def start_input_data_copy_time(self):
        self._start_timer(Metrics.INPUT_COPY_TIME, Metrics.INPUT_COPY_START_TIMESTAMP, Metrics.INPUT_COPY_END_TIMESTAMP)

    def stop_input_data_copy_time(self):
        return self._stop_timer(Metrics.INPUT_COPY_TIME)

    def start_output_data_copy_time(self):
        self._start_timer(Metrics.OUTPUT_COPY_TIME, Metrics.OUTPUT_COPY_START_TIMESTAMP, Metrics.OUTPUT_COPY_END_TIMESTAMP)

    def stop_output_data_copy_time(self):
        return self._stop_timer(Metrics.OUTPUT_COPY_TIME)

    def start_archive_packing_time(self):
        self._start_timer(
            Metrics.ARCHIVE_PACKING_TIME,
            Metrics.ARCHIVE_PACKING_START_TIMESTAMP,
            Metrics.ARCHIVE_PACKING_END_TIMESTAMP,
        )

    def stop_archive_packing_time(self):
        return self._stop_timer(Metrics.ARCHIVE_PACKING_TIME)

    def _start_timer(self, timespan_metric: Metrics, start_metric: Metrics, end_metric: Metrics):
        timer = Timer(start_metric, end_metric)
        self.update_metric_value(start_metric, timer.start_timestamp.isoformat())
        self.timers[timespan_metric] = timer

    def _stop_timer(self, timespan_metric: Metrics):
        timer = self.timers[timespan_metric]
        timer.stop_timer()
        self.update_metric_value(timer.end_metric, timer.end_timestamp.isoformat())
        self.update_metric_value(timespan_metric, timer.total_execution_time.total_seconds())

        return timer.total_execution_time
