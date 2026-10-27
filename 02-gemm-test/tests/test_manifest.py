from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manifest import load_manifest


class MatmulManifestTests(unittest.TestCase):
    def test_single_fp16_config(self) -> None:
        manifest = load_manifest(ROOT / "manifests" / "matmul_v1.json")
        self.assertEqual(manifest.config.compute_capability, "sm120")
        self.assertEqual(manifest.config.input_dtype, "float16")

    def test_tensor_core_tiles_meet_mma_shape(self) -> None:
        options = load_manifest(ROOT / "manifests" / "matmul_v1.json").config.options
        self.assertEqual(options["block_m"] % 16, 0)
        self.assertEqual(options["block_n"] % 16, 0)
        self.assertEqual(options["block_k"] % 16, 0)

    def test_all_three_tail_axes_are_present(self) -> None:
        manifest = load_manifest(ROOT / "manifests" / "matmul_v1.json")
        case = next(case for case in manifest.cases if case.case_id == "three-axis-tail")
        self.assertEqual((case.M, case.N, case.K), (37, 53, 29))

    def test_gau_benchmark_budget_is_fixed(self) -> None:
        manifest = load_manifest(ROOT / "manifests" / "matmul_v1.json")
        self.assertTrue(
            all((case.warmup_ms, case.rep_ms) == (25, 100) for case in manifest.cases)
        )
        timing = manifest.performance_policy["timing"]
        self.assertEqual(timing["method"], "triton.testing.do_bench")
        self.assertEqual(timing["return_mode"], "median")
        self.assertTrue(timing["clear_l2"])
        square = next(case for case in manifest.cases if case.case_id == "square-4096")
        self.assertEqual((square.M, square.N, square.K, square.pattern), (4096, 4096, 4096, "random"))


if __name__ == "__main__":
    unittest.main()
