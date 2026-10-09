import os
import unittest
from types import SimpleNamespace

import pandas as pd
try:
    import test_support  # noqa: F401
except ModuleNotFoundError:
    from MELD.tests import test_support  # noqa: F401


DATABASE_URL = os.environ.get("MELD_TEST_DATABASE_URL")


@unittest.skipUnless(DATABASE_URL, "set MELD_TEST_DATABASE_URL to run PostgreSQL integration tests")
class PostgreSQLIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sqlalchemy import create_engine, text

        cls.engine = create_engine(DATABASE_URL)
        with cls.engine.begin() as connection:
            connection.execute(text("CREATE TEMP TABLE meld_test_values (age INTEGER, label TEXT)"))
            connection.execute(text("INSERT INTO meld_test_values VALUES (1, 'one'), (NULL, 'two')"))

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_connection_and_parameterized_query(self):
        from InternalDataLoader import dataloader

        context = SimpleNamespace(
            contract=SimpleNamespace(
                input_schema=SimpleNamespace(
                    query=SimpleNamespace(statement="SELECT age, label FROM meld_test_values WHERE age >= :minimum"),
                    features=[SimpleNamespace(name="age", datatype="Int64"), SimpleNamespace(name="label", datatype="string")],
                )
            )
        )
        original_engine = dataloader.engine
        dataloader.engine = self.engine
        try:
            result = dataloader.execute_query(context, {"minimum": 1})
        finally:
            dataloader.engine = original_engine

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.to_dict("records"), [{"age": 1, "label": "one"}])

    def test_query_preserves_nulls_and_zero_rows(self):
        from InternalDataLoader import dataloader

        context = SimpleNamespace(
            contract=SimpleNamespace(
                input_schema=SimpleNamespace(
                    query=SimpleNamespace(statement="SELECT age, label FROM meld_test_values WHERE age IS NULL"),
                    features=[SimpleNamespace(name="age", datatype="Int64"), SimpleNamespace(name="label", datatype="string")],
                )
            )
        )
        original_engine = dataloader.engine
        dataloader.engine = self.engine
        try:
            result = dataloader.execute_query(context)
        finally:
            dataloader.engine = original_engine

        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["age"]))

    def test_sql_errors_are_propagated(self):
        from InternalDataLoader import dataloader

        context = SimpleNamespace(
            contract=SimpleNamespace(
                input_schema=SimpleNamespace(
                    query=SimpleNamespace(statement="SELECT definitely_missing_column FROM meld_test_values"),
                    features=[],
                )
            )
        )
        original_engine = dataloader.engine
        dataloader.engine = self.engine
        try:
            with self.assertRaises(Exception):
                dataloader.execute_query(context)
        finally:
            dataloader.engine = original_engine


if __name__ == "__main__":
    unittest.main()
