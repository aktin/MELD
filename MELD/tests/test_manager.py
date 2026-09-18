import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pandas as pd
import test_support

from ModelManager import Contract
from ModelManager import manager
from ModelEnvironment import ExecutionStatus


class ManagerTest(unittest.TestCase):
    def setUp(self):
        self.contract = Contract.from_dict(test_support.contract_data())
        self.context = MagicMock()
        self.context.contract = self.contract
        self.context.logger = MagicMock()
        self.context.cancel_requested = False

    def test_compute_time_window_supports_absolute_and_relative_scopes(self):
        self.contract.input_schema.temporal_scope.type = "absolute"
        self.contract.input_schema.temporal_scope.start = "2020-01-01T00:00:00+00:00"
        self.contract.input_schema.temporal_scope.end = "2020-01-02T00:00:00+00:00"
        start, end = manager._compute_time_window(self.context)
        self.assertEqual(start, datetime.fromisoformat("2020-01-01T00:00:00+00:00"))
        self.assertEqual(end, datetime.fromisoformat("2020-01-02T00:00:00+00:00"))

        self.contract.input_schema.temporal_scope.type = "relative"
        self.contract.input_schema.temporal_scope.value = "P1D"
        self.contract.input_schema.temporal_scope.anchor = "2020-01-02T00:00:00+00:00"
        start, end = manager._compute_time_window(self.context)
        self.assertEqual(end - start, timedelta(days=1))

    def test_compute_time_window_rejects_non_positive_windows(self):
        scope = self.contract.input_schema.temporal_scope
        scope.type = "absolute"
        scope.start = "2020-01-02T00:00:00+00:00"
        scope.end = "2020-01-01T00:00:00+00:00"

        with self.assertRaisesRegex(ValueError, "Start time must be before end time"):
            manager._compute_time_window(self.context)

    def test_query_data_updates_status_and_row_metric(self):
        result = pd.DataFrame({"age": [1, 2]})
        monitor = MagicMock()
        monitor.stop_query_execution_time.return_value = timedelta(seconds=1)
        with patch.object(manager, "execute_query", return_value=result) as execute_query:
            actual = manager.query_data(self.context, {"start": "start"}, monitor)

        self.assertIs(actual, result)
        execute_query.assert_called_once_with(self.context, {"start": "start"})
        self.assertEqual(
            self.context.set_status.call_args_list,
            [call(ExecutionStatus.START_QUERY), call(ExecutionStatus.QUERY_FINISHED)],
        )
        monitor.update_metric_value.assert_called_once()

    def test_query_data_marks_failure_and_re_raises(self):
        monitor = MagicMock()
        with patch.object(manager, "execute_query", side_effect=RuntimeError("database")):
            with self.assertRaisesRegex(RuntimeError, "database"):
                manager.query_data(self.context, {}, monitor)

        self.context.set_status.assert_any_call(ExecutionStatus.FAILED)
        monitor.stop_query_execution_time.assert_not_called()

    def test_validate_and_normalize_features_selects_contract_columns_and_converts_nulls(self):
        self.contract.input_schema.features = [
            type(self.contract.input_schema.features[0])("age", "Int64"),
            type(self.contract.input_schema.features[0])("score", "Float64"),
            type(self.contract.input_schema.features[0])("active", "boolean"),
            type(self.contract.input_schema.features[0])("label", "string"),
        ]
        df = pd.DataFrame({
            "age": pd.Series([1, None], dtype="float64"),
            "score": pd.Series([1.5, None], dtype="float64"),
            "active": pd.Series([True, None], dtype="object"),
            "label": pd.Series(["A", None], dtype="object"),
            "extra": [1, 2],
        })
        # Validation uses the source dtypes, so provide compatible columns first.
        df["age"] = df["age"].astype("Int64")
        df["active"] = df["active"].astype("boolean")
        features = manager._validate_features(df, self.context)
        normalized = manager._normalize_features(df, features)

        self.assertEqual([feature.name for feature in features], ["age", "score", "active", "label"])
        self.assertEqual(list(normalized.columns), ["age", "score", "active", "label"])
        self.assertEqual(str(normalized["age"].dtype), "int64")
        self.assertEqual(str(normalized["score"].dtype), "float32")
        self.assertEqual(str(normalized["active"].dtype), "int64")
        self.assertEqual(normalized["label"].tolist(), ["A", ""])

    def test_pack_metrics_creates_and_updates_zip_archive(self):
        monitor = MagicMock()
        monitor.collect_metrics.return_value = {"metric": 1}
        context = MagicMock(logger=MagicMock())
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/execution.zip"
            manager.pack_metrics(path, context, monitor)
            with zipfile.ZipFile(path) as archive:
                metrics = json.loads(archive.read("metrics.json"))

        self.assertEqual(metrics, {"metric": 1})
        monitor.collect_metrics.assert_called_once_with()

    def test_run_inference_stages_workflow_and_packs_metrics(self):
        self.context.output_data_path = "/tmp"
        data = pd.DataFrame({"age": pd.Series([1], dtype="int64")})
        monitor = MagicMock()
        with patch.object(manager, "ExecutionMonitor", return_value=monitor), patch.object(
            manager, "ensure_image_exists"
        ), patch.object(manager, "query_data", return_value=data), patch.object(
            manager, "run_runtime_inference"
        ) as runtime, patch.object(manager, "pack_metrics") as pack_metrics:
            manager.run_inference(self.contract, self.context)

        self.context.set_status.assert_any_call(ExecutionStatus.PREPARING)
        runtime.assert_called_once()
        pack_metrics.assert_called_once_with(
            "/tmp/summarized_execution.zip", self.context, monitor
        )
        monitor.stop_total_execution_time.assert_called_once_with()

    def test_pull_and_remove_runtime_use_contract_image_and_swallow_failures(self):
        with patch.object(manager, "pull_image") as pull, patch.object(manager, "delete_image") as delete:
            manager.pull_runtime(self.contract)
            manager.remove_runtime(self.contract)

        pull.assert_called_once_with("example/runtime:1.0.0@sha256:example")
        delete.assert_called_once_with("example/runtime:1.0.0@sha256:example")


if __name__ == "__main__":
    unittest.main()
