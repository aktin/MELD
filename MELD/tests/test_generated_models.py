import unittest
from dataclasses import is_dataclass
from pathlib import Path

import test_support  # noqa: F401

from ModelManager.generated import (
    OutputFileType,
    QueryType,
    OutputSchema,
    TemporalScopeType,
)
from ModelManager.generated.feature import Feature


class GeneratedModelsTest(unittest.TestCase):
    def test_generated_models_use_named_types_and_dataclasses(self):
        output_schema = OutputSchema(
            type=OutputFileType.csv,
            labels=[Feature(name="prediction", datatype="Float64")],
        )

        self.assertTrue(is_dataclass(output_schema))
        self.assertEqual(TemporalScopeType.relative.value, "relative")
        self.assertEqual(QueryType.sql.value, "sql")

    def test_generated_models_are_not_top_level_runtime_exports(self):
        import ModelManager

        self.assertFalse(hasattr(ModelManager, "SchemaOutputSchema"))
        self.assertFalse(hasattr(ModelManager, "ScheduleType"))

    def test_generated_models_do_not_import_pydantic(self):
        generated_path = Path(__file__).parents[1] / "ModelManager" / "generated"

        for model_path in generated_path.glob("*.py"):
            self.assertNotIn("pydantic", model_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
