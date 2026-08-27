from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matmul_add_manifest import load_matmul_add_manifest


class MatmulAddManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stages, self.cases = load_matmul_add_manifest(
            ROOT / "manifests" / "matmul_add_v1.json"
        )

    def test_fixed_stages_and_timing(self) -> None:
        self.assertEqual(
            tuple(self.stages),
            ("02b-baseline-fp32", "02b-fused-fp16", "02b-final"),
        )
        self.assertTrue(
            all((c.warmup, c.trials, c.samples) == (20, 5, 50) for c in self.cases)
        )

    def test_c_is_observable_and_tails_are_fixed(self) -> None:
        c_only = next(c for c in self.cases if c.case_id == "c-only")
        self.assertEqual((c_only.ab_pattern, c_only.c_pattern), ("zeros", "ramp"))
        tail = next(c for c in self.cases if c.case_id == "three-axis-tail")
        self.assertEqual((tail.M, tail.N, tail.K), (37, 53, 29))


if __name__ == "__main__":
    unittest.main()
