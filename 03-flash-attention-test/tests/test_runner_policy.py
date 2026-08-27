from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import runner


class RunnerPolicyTests(unittest.TestCase):
    def test_missing_core_baseline_blocks_with_nonzero_failure(self) -> None:
        case = SimpleNamespace(case_id="core-case", eligibility="core")
        output = io.StringIO()
        with patch.object(runner, "torch_varlen_available", return_value=False):
            with redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "core performance gate blocked"):
                    runner.run_performance(
                        [("student", None)],
                        [case],
                        {"thresholds": {}},
                        requested_stage="03c",
                    )
        self.assertIn('"status": "blocked"', output.getvalue())

    def test_missing_optional_baseline_is_na(self) -> None:
        case = SimpleNamespace(case_id="optional-case", eligibility="optional")
        output = io.StringIO()
        with patch.object(runner, "torch_varlen_available", return_value=False):
            with redirect_stdout(output):
                runner.run_performance(
                    [("student", None)],
                    [case],
                    {"thresholds": {}},
                    requested_stage="03c",
                )
        self.assertIn('"status": "N/A"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
