import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import test_support  # noqa: F401

from ExecutionMonitor import ExecutionMonitor, Metrics, Timer
from InternalDataLoader import dataloader


class MonitorTest(unittest.TestCase):
    def test_monitor_initializes_contract_metadata_and_collects_timers(self):
        contract = MagicMock(
            id="contract-id",
            contract=MagicMock(version="1.0.0"),
            input_schema=MagicMock(features=[1, 2]),
            output_schema=MagicMock(predictor=[1]),
            runtime=MagicMock(
                image=MagicMock(
                    tag="tag",
                    name="image",
                    digest="digest",
                )
            ),
        )
        provider = MagicMock()
        provider.get.return_value = MagicMock(execution_id="execution-id", contract=contract)
        monitor = ExecutionMonitor(provider)

        monitor.start_query_execution_time()
        elapsed = monitor.stop_query_execution_time()

        self.assertIsInstance(elapsed.total_seconds(), float)
        self.assertEqual(monitor.get_metric_value(Metrics.JOB_ID), "execution-id")
        self.assertEqual(monitor.get_metric_value(Metrics.EXPECTED_FEATURE_COUNT), 2)
        self.assertIn(Metrics.QUERY_EXECUTION_START_TIMESTAMP, monitor.collect_metrics())
        self.assertIn(Metrics.QUERY_EXECUTION_END_TIMESTAMP, monitor.collect_metrics())
        self.assertGreaterEqual(monitor.get_metric_value(Metrics.QUERY_EXECUTION_TIME), 0)

    def test_timer_records_elapsed_duration(self):
        timer = Timer(Metrics.INFERENCE_START_TIMESTAMP, Metrics.INFERENCE_END_TIMESTAMP)

        elapsed = timer.stop_timer()

        self.assertIs(timer.total_execution_time, elapsed)
        self.assertGreaterEqual(elapsed.total_seconds(), 0)


class DataLoaderTest(unittest.TestCase):
    def test_execute_query_passes_sql_params_and_feature_dtypes(self):
        context = MagicMock()
        context.contract.input_schema.query.statement = "SELECT age FROM patients"
        feature = MagicMock()
        feature.name = "age"
        feature.datatype = "Int64"
        context.contract.input_schema.features = [feature]
        result = pd.DataFrame({"age": [1]})
        connection = MagicMock()
        connection_manager = MagicMock()
        connection_manager.__enter__.return_value = connection

        with patch.object(dataloader.engine, "connect", return_value=connection_manager), patch(
            "InternalDataLoader.dataloader.pd.read_sql_query", return_value=result
        ) as read_sql:
            actual = dataloader.execute_query(context, {"start": "start"})

        self.assertIs(actual, result)
        read_sql.assert_called_once()
        self.assertEqual(read_sql.call_args.kwargs["params"], {"start": "start"})
        self.assertEqual(read_sql.call_args.kwargs["dtype"], {"age": "Int64"})
        self.assertEqual(str(read_sql.call_args.args[0]), "SELECT age FROM patients")


if __name__ == "__main__":
    unittest.main()
